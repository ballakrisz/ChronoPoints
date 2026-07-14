import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils    
    
    
def plot_trajectory_with_pcls(pts_sequence, seq_centroids, z_trajectory, poly_order=3, seq_idx=0, num_curve_points=50):
    """
    Args:
        pts_sequence:   (B, T, N, 3) - original point clouds
        seq_centroids:  (B, T, 6) - centroids with velocities
        z_trajectory:   (B, 12) - polynomial coefficients per coordinate (3rd order)
        poly_order:     order of polynomial (3)
        seq_idx:        which sequence in batch to plot
        num_curve_points: number of points to sample along the polynomial curve
    """
    centroids = seq_centroids[seq_idx].cpu().numpy()
    coeffs = z_trajectory[seq_idx].cpu().numpy()  # (12,)
    pts_seq = pts_sequence[seq_idx].cpu().numpy()  # (T, N, 3)
    T = centroids.shape[0]
    
    # Color map
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(T)]
    
    # Polynomial coefficients
    coeffs_x = coeffs[0:poly_order+1]
    coeffs_y = coeffs[poly_order+1:2*(poly_order+1)]
    coeffs_z = coeffs[2*(poly_order+1):]
    
    # Sample curve
    t_fine = np.linspace(0, T-1, num_curve_points)
    x_curve = sum(c * t_fine**i for i, c in enumerate(coeffs_x))
    y_curve = sum(c * t_fine**i for i, c in enumerate(coeffs_y))
    z_curve = sum(c * t_fine**i for i, c in enumerate(coeffs_z))
    
    # Plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot PCL points per frame
    for t in range(T):
        ax.scatter(pts_seq[t,:,0], pts_seq[t,:,1], pts_seq[t,:,2], color=colors[t], alpha=0.3, label=f'Frame {t}')
    
    # Plot centroids per frame (same color as points)
    for t in range(T):
        ax.scatter(centroids[t,0], centroids[t,1], centroids[t,2], color=colors[t], marker='x', s=100)
    
    # Plot polynomial trajectory
    ax.plot(x_curve, y_curve, z_curve, color='black', linewidth=2, label='Fitted curve')
    
    # Equal axis scaling
    all_points = np.concatenate([pts_seq.reshape(-1,3), centroids[:,:3]], axis=0)
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    
    
# for 3 encoders
class CrossAttentionFusion(nn.Module):
    def __init__(self, emb_dim, num_heads=4):
        super().__init__()

        # cross-attention layers
        self.attn_traj = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, batch_first=False
        )
        self.attn_dist = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, batch_first=False
        )
        
        self.attn_spatio = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, batch_first=False
        )

        self.norm_traj = nn.LayerNorm(emb_dim)
        self.norm_dist = nn.LayerNorm(emb_dim)
        self.norm_spatio = nn.LayerNorm(emb_dim)

    def forward(self, E_traj, E_dist, E_spatio):
        """
        E_traj:   (B, D)  trajectory embedding
        E_dist:   (B, D)  distortion embedding
        E_spatio: (B, D)  spatiotemporal local-global embedding
        """

        B, D = E_traj.shape

        # Convert to (L=1, B, D) for MHA
        T  = E_traj.unsqueeze(0)
        Dd = E_dist.unsqueeze(0)
        S  = E_spatio.unsqueeze(0)

        # ------------------------------------------------------------------
        # Trajectory attends to distortion + spatio
        # ------------------------------------------------------------------
        KV_traj = torch.cat([Dd, S], dim=0)  # (2, B, D)
        F_traj, _ = self.attn_traj(T, KV_traj, KV_traj)
        F_traj = self.norm_traj(F_traj + T)

        # ------------------------------------------------------------------
        # Distortion attends to trajectory + spatio
        # ------------------------------------------------------------------
        KV_dist = torch.cat([T, S], dim=0)
        F_dist, _ = self.attn_dist(Dd, KV_dist, KV_dist)
        F_dist = self.norm_dist(F_dist + Dd)

        # ------------------------------------------------------------------
        # Spatiotemporal attends to trajectory + distortion
        # ------------------------------------------------------------------
        KV_spatio = torch.cat([T, Dd], dim=0)
        F_spatio, _ = self.attn_spatio(S, KV_spatio, KV_spatio)
        F_spatio = self.norm_spatio(F_spatio + S)

        # Back to (B, D)
        F_traj   = F_traj.squeeze(0)
        F_dist   = F_dist.squeeze(0)
        F_spatio = F_spatio.squeeze(0)

        # ------------------------------------------------------------------
        # Final fusion
        # ------------------------------------------------------------------
        F = torch.cat(
            [F_traj, F_dist, F_spatio],
            dim=-1
        )  # (B, 3D)

        return F


class GatedFusion(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 3)
        )

    def forward(self, z_dist, z_traj, z_spatio):
        z_all = torch.cat([z_traj, z_dist, z_spatio], dim=-1)  # (B, 3D)
        weights = torch.softmax(self.gate(z_all), dim=-1)      # (B, 3)

        z_fused = (
            weights[:, 0:1] * z_traj +
            weights[:, 1:2] * z_dist +
            weights[:, 2:3] * z_spatio
        )

        return z_fused
# Also add this to the classifier, extended with zeroing all 3 modalities, so 0-0.2:traj, 0.2-0.4:dist, 0.4-0.6:spatiotemp
# if self.training:
    # if torch.rand(1) < 0.2:
        # z_traj = torch.zeros_like(z_traj)
        
class CrossEncoderContrastiveLoss(nn.Module): 
    """ 3-way symmetric InfoNCE across encoders. """ 
    def __init__(self, temperature=0.07): 
        super().__init__() 
        self.temperature = temperature 
        
    def _pairwise_loss(self, z1, z2): 
        """ Symmetric InfoNCE between two embedding sets. """ 
        B = z1.shape[0] 
        z1 = F.normalize(z1, dim=1) 
        z2 = F.normalize(z2, dim=1) 
        logits = torch.matmul(z1, z2.T) / self.temperature 
        labels = torch.arange(B, device=z1.device) 
        loss_12 = F.cross_entropy(logits, labels) 
        loss_21 = F.cross_entropy(logits.T, labels) 
        return (loss_12 + loss_21) * 0.5 
    
    def forward(self, p_dist=None, p_traj=None, p_spatio=None): 
        losses = [] 
        if p_dist is not None and p_traj is not None: 
            losses.append(self._pairwise_loss(p_dist, p_traj)) 
        if p_dist is not None and p_spatio is not None: 
            losses.append(self._pairwise_loss(p_dist, p_spatio)) 
        if p_traj is not None and p_spatio is not None: 
            losses.append(self._pairwise_loss(p_traj, p_spatio)) 
        if len(losses) == 0: 
            raise ValueError("At least two modalities must be provided") 
        return torch.stack(losses).mean()
        
# TODO: add a memory bank or negative dictionra, batch size of 32 is waay too low
class CrossEncoderContrastiveLossWithQueue(nn.Module):
    """
    3-way symmetric InfoNCE with memory queues (MoCo-style).
    """

    def __init__(self, temperature=0.07, queue_size=32, emb_dim=128):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # ---- memory queues (one per modality) ----
        self.register_buffer("queue_dist", torch.randn(queue_size, emb_dim))
        self.register_buffer("queue_traj", torch.randn(queue_size, emb_dim))
        self.register_buffer("queue_spatio", torch.randn(queue_size, emb_dim))

        self.queue_dist = F.normalize(self.queue_dist, dim=1)
        self.queue_traj = F.normalize(self.queue_traj, dim=1)
        self.queue_spatio = F.normalize(self.queue_spatio, dim=1)

        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, queue, z):
        B = z.shape[0]
        ptr = int(self.ptr)

        if ptr + B <= self.queue_size:
            queue[ptr:ptr + B] = z
        else:
            first = self.queue_size - ptr
            queue[ptr:] = z[:first]
            queue[:B - first] = z[first:]

    @torch.no_grad()
    def _update_queues(self, p_dist, p_traj, p_spatio):
        B = p_dist.shape[0]
        ptr = int(self.ptr)

        self._enqueue(self.queue_dist, p_dist)
        self._enqueue(self.queue_traj, p_traj)
        self._enqueue(self.queue_spatio, p_spatio)

        self.ptr[0] = (ptr + B) % self.queue_size

    def _pairwise_loss(self, z1, z2, queue2):
        """
        Symmetric InfoNCE with memory queue.
        """
        B = z1.shape[0]

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # ---- positives ----
        logits_pos = torch.sum(z1 * z2, dim=1, keepdim=True)  # (B, 1)

        # ---- negatives from queue ----
        queue2 = queue2.detach().clone()
        logits_neg = torch.matmul(z1, queue2.T)  # (B, K)

        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature

        labels = torch.zeros(B, dtype=torch.long, device=z1.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, p_dist=None, p_traj=None, p_spatio=None):

        losses = []

        if p_dist is not None and p_traj is not None:
            losses.append(self._pairwise_loss(p_dist, p_traj, self.queue_traj))
            losses.append(self._pairwise_loss(p_traj, p_dist, self.queue_dist))

        if p_dist is not None and p_spatio is not None:
            losses.append(self._pairwise_loss(p_dist, p_spatio, self.queue_spatio))
            losses.append(self._pairwise_loss(p_spatio, p_dist, self.queue_dist))

        if p_traj is not None and p_spatio is not None:
            losses.append(self._pairwise_loss(p_traj, p_spatio, self.queue_spatio))
            losses.append(self._pairwise_loss(p_spatio, p_traj, self.queue_traj))

        if len(losses) == 0:
            raise ValueError("At least two modalities must be provided")

        loss = torch.stack(losses).mean()

        # ---- update memory AFTER loss ----
        with torch.no_grad():
            self._update_queues(
                F.normalize(p_dist, dim=1).detach(),
                F.normalize(p_traj, dim=1).detach(),
                F.normalize(p_spatio, dim=1).detach(),
            )

        return loss

        
class DistortionSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss applied ONLY on z_distortion. 
    Uses cosine similarity internally for computing similarity between feature vectors.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D_dist)  -- z_distortion
        labels:   (B,)
        """
        device = features.device

        # Normalize for cosine similarity
        features = F.normalize(features, dim=1)

        B = features.size(0)
        sim = torch.matmul(features, features.T) / self.temperature  # (B,B)

        # label mask: positives=1 for same-class
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # mask-out self comparisons
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss
    
    
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        features: (B, D)
        labels:   (B,)
        """
        device = features.device
        features = F.normalize(features, dim=1)

        B = features.size(0)
        sim = torch.matmul(features, features.T) / self.temperature  # (B,B)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # no self-comparison
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss
    
    
class CrossViewSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss between two different embeddings:
    z_distortion  <->  z_trajectory

    Encourages embeddings of the *same class* to be close across views,
    and different classes to be farther apart.

    Args:
        temperature (float): temperature scaling for similarity scores.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_distortion, z_trajectory, labels):
        """
        z_distortion: (B, D1)  -- embedding 1
        z_trajectory: (B, D2)  -- embedding 2
        labels:       (B,)
        """

        device = z_distortion.device

        # Normalize both embeddings
        zd = F.normalize(z_distortion, dim=1)
        zt = F.normalize(z_trajectory, dim=1)

        # Cross-view cosine similarity
        # zd (B,D1) × zt.T (D2,B) -> (B,B)
        sim = torch.matmul(zd, zt.T) / self.temperature

        # Build positive mask (same class = 1)
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float().to(device)

        # Avoid log(0)
        log_prob = F.log_softmax(sim, dim=1)

        # Positive log-likelihood only
        pos_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        loss = -pos_log_prob.mean()
        return loss
    

def farthest_point_sample(xyz, npoint):
    """
    xyz: (B, N, 3)
    return: (B, npoint)
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]

    return centroids

def make_radius_table(r_min, r_max, num_frames, device):
    """
    EXACT equivalent of:
    np.linspace(r_min, r_max, num_frames)
    """
    return torch.linspace(r_min, r_max, steps=num_frames, device=device)


def temporal_radius_from_dt(delta_t, radius_table):
    """
    delta_t: (...,) float or int
    radius_table: (T,)
    """
    delta_t = delta_t.long().clamp(0, radius_table.shape[0] - 1)
    return radius_table[delta_t]



def spatiotemporal_group_flat_radius(
    xyz,
    time,
    feats,
    anchor_idx,
    nsample,
    radius_table
):
    """
    Hybrid CUDA implementation:
    - PointNet++ ball query for spatial candidates
    - temporal adaptive filtering afterward

    Preserves original behavior approximately,
    but MUCH faster than dense cdist.
    """

    B, N, _ = xyz.shape
    K = anchor_idx.shape[1]

    # =========================================================
    # Gather anchor xyz/time
    # =========================================================

    anchor_xyz = pointnet2_utils.gather_operation(
        xyz.transpose(1, 2).contiguous(),
        anchor_idx
    ).transpose(1, 2).contiguous()  # (B,K,3)

    anchor_time = pointnet2_utils.gather_operation(
        time.transpose(1, 2).contiguous(),
        anchor_idx
    ).transpose(1, 2).contiguous()  # (B,K,1)

    # =========================================================
    # Use MAX radius for spatial preselection
    # =========================================================

    max_radius = radius_table.max().item()

    idx = pointnet2_utils.ball_query(
        max_radius,
        nsample,
        xyz.contiguous(),
        anchor_xyz.contiguous()
    ).int()  # (B,K,nsample)

    # =========================================================
    # Gather neighbor xyz
    # =========================================================

    grouped_xyz = pointnet2_utils.grouping_operation(
        xyz.transpose(1, 2).contiguous(),
        idx
    )

    grouped_xyz = grouped_xyz.permute(
        0, 2, 3, 1
    ).contiguous()  # (B,K,nsample,3)

    # =========================================================
    # Gather neighbor time
    # =========================================================

    grouped_time = pointnet2_utils.grouping_operation(
        time.transpose(1, 2).contiguous(),
        idx
    )

    grouped_time = grouped_time.permute(
        0, 2, 3, 1
    ).contiguous()  # (B,K,nsample,1)

    # =========================================================
    # Temporal adaptive radius filtering
    # =========================================================

    delta_t = torch.abs(
        grouped_time - anchor_time.unsqueeze(2)
    )  # (B,K,nsample,1)

    adaptive_radius = temporal_radius_from_dt(
        delta_t.squeeze(-1),
        radius_table
    )  # (B,K,nsample)

    spatial_dist = torch.norm(
        grouped_xyz - anchor_xyz.unsqueeze(2),
        dim=-1
    )  # (B,K,nsample)

    valid_mask = spatial_dist <= adaptive_radius

    # =========================================================
    # Relative xyz
    # =========================================================

    rel_xyz = grouped_xyz - anchor_xyz.unsqueeze(2)

    # zero invalid neighbors
    rel_xyz = rel_xyz * valid_mask.unsqueeze(-1)

    # =========================================================
    # Features
    # =========================================================

    if feats is not None:

        grouped_feats = pointnet2_utils.grouping_operation(
            feats.transpose(1, 2).contiguous(),
            idx
        )

        grouped_feats = grouped_feats.permute(
            0, 2, 3, 1
        ).contiguous()

        grouped_feats = grouped_feats * valid_mask.unsqueeze(-1)

        return torch.cat(
            [rel_xyz, grouped_feats],
            dim=-1
        )

    return rel_xyz



