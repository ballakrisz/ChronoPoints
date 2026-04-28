import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from chronopoints_utils import farthest_point_sample, spatiotemporal_group_flat_radius, make_radius_table, CrossAttentionFusion, GatedFusion
import torch
import torch.nn as nn
import torch.nn.functional as F
    
    

def compute_centroid(pc, mask=None, mode="mean", eps=1e-6):
    """
    Compute the centroid of a point cloud using either:
      - mean of valid points
      - center of bounding box of valid points

    Args:
        pc:   (B, T, N, 3)
        mask: (B, T, N) boolean mask
        mode: "mean" | "bbox"

    Returns:
        centroid: (B, T, 3)
    """
    if mask is not None:
        valid = mask.float()               # (B,T,N)
        valid_exp = valid.unsqueeze(-1)    # (B,T,N,1)

        if mode == "mean":
            denom = valid.sum(dim=2, keepdim=True) + eps      # (B,T,1)
            centroid = (pc * valid_exp).sum(dim=2) / denom    # (B,T,3)

        elif mode == "bbox":
            big = 1e9
            pc_min = pc.clone()
            pc_max = pc.clone()
            pc_min[valid == 0] =  big
            pc_max[valid == 0] = -big

            min_xyz = pc_min.min(dim=2).values                # (B,T,3)
            max_xyz = pc_max.max(dim=2).values                # (B,T,3)
            centroid = (min_xyz + max_xyz) / 2.0

        else:
            raise ValueError(f"Unknown mode '{mode}' for centroid extraction")

    else:
        # mask not provided
        if mode == "mean":
            centroid = pc.mean(dim=2)                         # (B, T, 3)
        elif mode == "bbox":
            min_xyz = pc.min(dim=2).values
            max_xyz = pc.max(dim=2).values
            centroid = (min_xyz + max_xyz) / 2.0
        else:
            raise ValueError(f"Unknown mode '{mode}' for centroid extraction")

    return centroid


# ---------- Rigid Alignment Utility ----------
def align_frames(pc, mask=None, centroid_mode="mean"):
    """
    Removes translation per frame by subtracting the chosen centroid.

    Args:
        pc:   (B, T, N, 3)
        mask: (B, T, N)
        mode: "mean" or "bbox"

    Returns:
        aligned_pc: (B, T, N, 3)
    """
    centroid = compute_centroid(pc, mask=mask, mode=centroid_mode)  # (B,T,3)
    aligned_pc = pc - centroid.unsqueeze(2)                # (B,T,N,3)
    return aligned_pc


# ---------- Frame Encoder ----------
class FrameEncoder(nn.Module):
    """
    Encodes each frame's sparse point cloud into a fixed-size feature vector.
    
    Motivation:
    - Point clouds are unordered; order of points carries no meaning.
    - Therefore, we use a shared MLP applied to each point individually
      (a "pointwise" embedding), followed by a symmetric pooling operation
      (max) to achieve permutation invariance.
    """

    def __init__(self, in_channels=3, hidden_dims=(64,128), out_dim=128):
        super().__init__()

        # Build a simple shared MLP: [in -> hidden -> ... -> out]
        layers = []
        last = in_channels
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        
        self.post_pool_dropout = nn.Dropout(p=0.1)

    def forward(self, pts, mask=None):
        """
        Args:
            pts:  (B, T, N, 3) - batch of sequences  of point clouds belonging to the same sequence
            mask: (B, T, N) - optional boolean mask (True=valid, False=padded), indicating actual and padded points

        Returns:
            frame_features: (B, T, D) - embedding per frame
        """
        B, T, N, C = pts.shape

        # Flatten batch and time for efficient processing
        x = pts.view(B*T, N, C)

        # Shared MLP on each point independently
        x = self.mlp(x)  # (B*T, N, D)

        # Mask invalid (padded) points if present
        if mask is not None:
            m = mask.view(B * T, N)          # (B*T, N)
            neg_inf = torch.finfo(x.dtype).min / 2
            x = x.masked_fill(~m.unsqueeze(-1), neg_inf)

        # Global max pooling over points (permutation-invariant aggregation)
        x = x.max(dim=1).values  # (B*T, D)
        # x = self.post_pool_dropout(x) # don't use dropout in the shared MLP -> too destructive
        
        if mask is not None:
            # detect frames with no valid points
            frame_has_points = mask.view(B * T, N).any(dim=1)  # (B*T,)
            # zero-out features where there were no points at all
            x[~frame_has_points] = 0.0

        # Reshape back to sequence form
        return x.view(B, T, -1)  # (B, T, D)


# ---------- Distortion Encoder (with FrameEncoder inside) ----------
class DistortionEncoder(nn.Module):
    """
    Learns a latent representation of *non-rigid shape distortions* across a point cloud sequence.

    Design motivation:
    - Birds flap and deform more than UAVs → intra-frame shape changes encode class information.
    - We therefore learn to represent *how* each frame's embedding changes over time.
    
    Design:
      1. Translation normalization (remove global motion, align frames)  --> we only care about distortion, not rigid motion
      2. FrameEncoder for per-frame latents, that embed the distortion of the point cloud's shape.
      3. Compute Δf = f_{t+1} - f_t to represent shape changes
      4. Temporal modeling via Transformer to learn temporal dependencies in distortions
      5. Pool over time → global z_distortion latent for the sequence
    """

    def __init__(self, point_features=3, frame_enc_dims=(64,128), emb_dim=128, transformer_heads=4, transformer_layers=2,centroid_mode="mean"):
        super().__init__()
        self.emb_dim = emb_dim
        self.temporal_dropout = nn.Dropout(p=0.1)
        self.centroid_mode = centroid_mode

        # --- (2) Spatial encoding ---
        self.frame_enc = FrameEncoder(
            in_channels=point_features,
            hidden_dims=frame_enc_dims,
            out_dim=emb_dim
        )

        # --- (4) Temporal modeling block (Transformer) ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=transformer_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
            activation='gelu'
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        
        # --- (5) Final projection to latent distortion embedding ---
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, pts, mask=None):
        """
        Args:
            pts:        (B, T, N, 3) - sequence of point clouds
            mask:       (B, T, N) - valid point mask
        Returns:
            z_distortion: (B, D) - sequence-level latent capturing distortion
        """
        B, T, N, _ = pts.shape
        if T < 2:
            raise ValueError("Need at least 2 frames to compute deltas")
        
        # frame-level validity: True if any point is valid
        if mask is not None:
            frame_valid = (mask.sum(dim=2) > 0)  # (B, T)
        else:
            frame_valid = torch.ones(B, T, dtype=torch.bool, device=pts.device)
        
        # --- 1. Align each frame to remove rigid motion ---
        aligned_pts = align_frames(pts, mask=mask, centroid_mode=self.centroid_mode)  # (B, T, N, 3)

        # --- 2. Encode each frame spatially ----
        f_seq = self.frame_enc(aligned_pts, mask=mask)  # (B, T, D)

        # --- 3. Compute temporal feature deltas ----
        # deltas only make sense where both neighbored frames are valid
        delta_f = f_seq[:, 1:, :] - f_seq[:, :-1, :]  # (B, T-1, D)
        delta_valid = frame_valid[:, 1:] & frame_valid[:, :-1]  # (B, T-1)

        # Transformer masking: True = pad/ignore
        src_key_padding_mask = ~delta_valid  # (B, T-1)

        out = self.temporal(delta_f, src_key_padding_mask=src_key_padding_mask)  # (B, T-1, D)
        # out = self.temporal_dropout(out)

        # masked temporal average over valid deltas
        delta_valid_f = delta_valid.float()             # (B, T-1)
        denom = delta_valid_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        pooled = (out * delta_valid_f.unsqueeze(-1)).sum(dim=1) / denom  # (B, D)

        z_distortion = self.proj(pooled)  # (B, D)
        return z_distortion


# ---------- trajectory utility function ----------
def extract_centroids_with_velocities(pts, mask, velocities, eps=1e-6, centroid_mode="mean"):
    """
    Extract per-frame centroids and their velocities from a sequence of point clouds.
    
    Args:
        pts: (B, T, N, 3)
        mask: (B, T, N) - valid point mask
        velocities: (B, T, 3) - point velocities
    Returns:
        seq_centroids: (B, T, 6) - per-frame centroids with x,y,z velocities
    """
    B, T, N, _ = pts.shape

    # IMPORTANT CONSIDERATION: The centroids are normalized, but the velocities are NOT, they are in m/s format.

    # --- 1. Compute centroids ---
    valid = mask.float()
    centroid = compute_centroid(pts, mask=mask, mode=centroid_mode)  # (B, T, 3)

    # frame-level validity
    frame_valid = (valid.sum(dim=2) > 0)              # (B, T)

    # sanitize velocities (in case of NaNs / Infs) -> though its highly unlikely, as dataloader should prevent this (lots of sanity checks)
    velocities = torch.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)

    # explicitly zero velocities where frame is invalid
    velocities = velocities * frame_valid.unsqueeze(-1)  # (B, T, 3)

    seq_centroids = torch.cat([centroid, velocities], dim=-1)  # (B, T, 6)
    return seq_centroids, frame_valid


    
# ---------- Trajectory Encoder ----------
class TrajectoryEncoder(nn.Module):
    def __init__(self, T, emb_dim=128, hidden_dims=(256, 128), poly_order=3, centroid_mode="mean"):
        """
        Args:
            T: number of frames per sequence
            emb_dim: output latent dimension
            hidden_dims: tuple of hidden layer sizes
            poly_order: order of polynomial used for trajectory fitting
        """
        super(TrajectoryEncoder, self).__init__()
        self.T = T
        print(f"initialized Trajectory encoder with polynomial order: {poly_order}")
        self.poly_order = poly_order
        self.input_dim = T * 6 + 3*(poly_order+1)  # 6 features per frame + polynomial coefficients
        self.centroid_mode = centroid_mode
        
        layers = []
        last_dim = self.input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.2))
            last_dim = h
            
        layers.append(nn.Linear(last_dim, emb_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5)) 
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, pts, mask, velocities):
        """
        Args:
            pts:        (B, T, N, 3) - sequence of point clouds
            mask:       (B, T, N) - valid point mask
            velocities:       (B, T, N, 3) - point velocities
        Returns:
            z_trajectory: (B, D) - sequence-level latent capturing trajectory
        """
        
        seq_centroids, frame_valid = extract_centroids_with_velocities(pts, mask=mask, velocities=velocities, centroid_mode=self.centroid_mode)  # (B, T, 6)
        
        B, T, F = seq_centroids.shape
        assert T == self.T, "Input sequence length does not match TrajectoryEncoder initialization."
        
        # time index
        t = torch.arange(T, dtype=seq_centroids.dtype, device=seq_centroids.device)  # (T,)
        A = torch.stack([t ** i for i in range(self.poly_order + 1)], dim=-1)       # (T, order+1)
        A = A.unsqueeze(0).expand(B, -1, -1)                                       # (B, T, order+1)

        # weights: 1 for valid frame, 0 for empty frame
        w = frame_valid.float()  # (B, T)

        polyn_coeffs = []
        for dim in range(3):  # fit poly for x,y,z
            y = seq_centroids[:, :, dim]  # (B, T)

            # Weighted least squares: multiply rows by weight
            w_exp = w.unsqueeze(-1)             # (B, T, 1)
            A_w = A * w_exp                     # (B, T, order+1)
            y_w = y.unsqueeze(-1) * w_exp       # (B, T, 1)

            # stable batched least squares
            coeffs = torch.linalg.lstsq(A_w, y_w).solution  # (B, order+1, 1)
            polyn_coeffs.append(coeffs.squeeze(-1))         # (B, order+1)

        polyn_coeffs = torch.cat(polyn_coeffs, dim=-1)      # (B, 3*(order+1))

        # flatten centroids (empty frames contribute zeros)
        seq_flat = seq_centroids.view(B, -1)                # (B, T*6)

        x = torch.cat([seq_flat, polyn_coeffs], dim=-1)     # (B, T*6 + 3*(order+1))

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z_trajectory = self.mlp(x)                          # (B, emb_dim)

        # You can visualize the sequence and fitted trajectory for debugging
        # plot_trajectory_with_pcls(
        #     pts_sequence=pts, 
        #     seq_centroids=seq_centroids, 
        #     z_trajectory=polyn_coeffs, 
        #     seq_idx=0,
        #     poly_order=self.poly_order
        # )
        return z_trajectory
    

class SpatioTemporalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, npoint, nsample, radius_table):
        super().__init__()

        self.npoint = npoint
        self.nsample = nsample
        self.register_buffer("radius_table", radius_table)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, xyz, time, feats):
        """
        xyz:   (B, N, 3)
        time:  (B, N, 1)
        feats: (B, N, C)
        """

        # Farthest point sampling
        anchor_idx = farthest_point_sample(xyz, self.npoint)

        # Spatio-temporal grouping
        grouped = spatiotemporal_group_flat_radius(
            xyz,
            time,
            feats,
            anchor_idx,
            self.nsample,
            self.radius_table
        )
        # grouped: (B, npoint, nsample, C+3)

        # Shared MLP
        x = self.mlp(grouped)

        # Max pool over neighbors
        x = x.max(dim=2).values  # (B, npoint, out_channels)

        # Gather new xyz + time
        new_xyz = torch.gather(
            xyz, 1, anchor_idx.unsqueeze(-1).expand(-1, -1, 3)
        )

        new_time = torch.gather(
            time, 1, anchor_idx.unsqueeze(-1)
        )

        return new_xyz, new_time, x


# ------------------------------------------------------------
# MeteorNet-Like Encoder
# ------------------------------------------------------------
class SpatioTemporalEncoder(nn.Module):
    def __init__(self, emb_dim=128, num_frames=12):
        super().__init__()

        base_radius = torch.linspace(0.2, 0.6, num_frames)

        # ---- ONLY ONE ST LAYER ----
        self.sa = SpatioTemporalLayer(
            in_channels=1,       # using time as feature
            out_channels=64,     # MUCH smaller
            npoint=512,          # fewer anchors
            nsample=16,          # fewer neighbors
            radius_table=base_radius
        )

        # ---- Small projection head ----
        self.proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),      # regularization
            nn.Linear(128, emb_dim)
        )

    def forward(self, pts):
        """
        pts: (B, T, N, 3)
        """

        B, T, N, _ = pts.shape
        device = pts.device

        xyz = pts.reshape(B, T * N, 3)

        time = (
            torch.arange(T, device=device)
            .view(1, T, 1)
            .expand(B, T, N)
            .reshape(B, T * N, 1)
            .float()
        )

        # ---- Single abstraction ----
        xyz, time, features = self.sa(xyz, time, time)
        # (B, npoint, 64)

        # ---- Global pooling over anchors ----
        features = features.transpose(1, 2)   # (B, 64, npoint)
        pooled = features.max(dim=2).values   # (B, 64)

        embedding = self.proj(pooled)

        return embedding


class ProjectionHead(nn.Module):
    """
    2-layer projection head used ONLY for contrastive learning.
    """

    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class ChronoPointsClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        emb_dim=128,
        traj_T=10,
        fusion_hidden_factor=4,
        fusion_dropout=0.2,
        centroid_mode="mean",
        poly_order=3
    ):
        super().__init__()

        ''' ---- Distortion Encoder ---- '''
        self.distortion_encoder = DistortionEncoder(
            point_features=3,
            frame_enc_dims=(64, 128),
            emb_dim=emb_dim,
            transformer_heads=4,
            transformer_layers=2,
            centroid_mode=centroid_mode
        )
        self.proj_dist = ProjectionHead(emb_dim, proj_dim=128)

        ''' ---- Trajectory Encoder ---- '''
        self.trajectory_encoder = TrajectoryEncoder(
            T=traj_T,
            emb_dim=emb_dim,
            hidden_dims=(256, 128),
            poly_order=poly_order,
            centroid_mode=centroid_mode
        )
        self.proj_traj = ProjectionHead(emb_dim, proj_dim=128)

        ''' ---- SpatioTemproal Encoder ---- '''
        self.spatiotemporal_encoder = SpatioTemporalEncoder(
            emb_dim=emb_dim,
            num_frames=traj_T
        )
        self.proj_spatio = ProjectionHead(emb_dim, proj_dim=128)

        # ---- 3-way learnable ensemble weights ----
        self.fusion_logits = nn.Parameter(torch.zeros(3))
        fusion_dim = emb_dim  # weighted sum keeps dimension unchanged

        # ---- MLP classifier head ----
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * fusion_hidden_factor),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),

            nn.Linear(fusion_dim * fusion_hidden_factor, emb_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),

            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, pts, mask, velocities):
        """
        Args:
            pts:        (B, T, N, 3)
            mask:       (B, T, N)
            velocities: (B, T, 3)
        Returns:
            logits: (B, num_classes)
        """

        # ---- Encode modalities ----
        z_dist = self.distortion_encoder(pts, mask=mask)
        z_traj = self.trajectory_encoder(pts, mask=mask, velocities=velocities)
        z_spatio = self.spatiotemporal_encoder(pts)

        # ---- Projection space (for contrastive loss only) ----
        p_dist = self.proj_dist(z_dist)
        p_traj = self.proj_traj(z_traj)
        p_spatio = self.proj_spatio(z_spatio)
        
        
        # ---- 3-way ensemble fusion ----
        w = torch.softmax(self.fusion_logits, dim=0)  # (3,)

        z_fused = (
            w[0] * z_dist +
            w[1] * z_traj +
            w[2] * z_spatio
        )  # (B, D)
        
        
        # ---- Classification ----
        logits = self.classifier(z_fused)

        return logits, (p_dist, p_traj, p_spatio)
