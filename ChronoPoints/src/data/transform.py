import numpy as np

def rotate_sequence(points_seq, max_angle=np.pi / 6):
    # Bias toward Z (radar azimuth invariance)
    axis = np.random.choice(['z', 'z', 'z', 'x', 'y'])
    theta = np.random.uniform(-max_angle, max_angle)

    c, s = np.cos(theta), np.sin(theta)

    if axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == 'x':
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    out = []
    for pc in points_seq:
        xyz = pc[:, :3] @ R.T
        out.append(np.hstack([xyz, pc[:, 3:]]) if pc.shape[1] > 3 else xyz)

    return out

def anisotropic_scale_sequence(points_seq, scale_range=(0.85, 1.15)):
    scales = np.random.uniform(*scale_range, size=3)

    out = []
    for pc in points_seq:
        xyz = pc[:, :3] * scales
        out.append(np.hstack([xyz, pc[:, 3:]]) if pc.shape[1] > 3 else xyz)
    return out

def scale_sequence(points_seq, scale_range=(0.80, 1.25)):
    """
    Uniformly scale XYZ coordinates only. Other features remain unchanged.
    """
    scale = np.random.uniform(*scale_range) # consistent for the sequence
    scaled_seq = []
    for pc in points_seq:
        coords = pc[:, :3]
        other_feats = pc[:, 3:] if pc.shape[1] > 3 else None
        coords *= scale
        if other_feats is not None:
            pc_aug = np.hstack([coords, other_feats])
        else:
            pc_aug = coords
        scaled_seq.append(pc_aug)
    return scaled_seq

def shift_sequence(points_seq, shift_range=0.1):
    """
    Apply a random translation to the entire sequence.
    """
    # Random shift along x, y, z
    shift = np.random.uniform(-shift_range, shift_range, size=(3,)) # consistent for the sequence
    
    shifted_seq = []
    for pc in points_seq:
        coords = pc[:, :3]  # XYZ
        other_feats = pc[:, 3:] if pc.shape[1] > 3 else None
        coords = coords + shift
        if other_feats is not None:
            pc_aug = np.hstack([coords, other_feats])
        else:
            pc_aug = coords
        shifted_seq.append(pc_aug)
    
    return shifted_seq

def jitter_sequence(points_seq, sigma=0.01, temporal_smooth=0.5):
    noise_prev = np.zeros_like(points_seq[0][:, :3])

    out = []
    for pc in points_seq:
        noise = (
            temporal_smooth * noise_prev
            + (1 - temporal_smooth) * np.random.normal(0, sigma, pc[:, :3].shape)
        )
        xyz = pc[:, :3] + noise
        out.append(np.hstack([xyz, pc[:, 3:]]) if pc.shape[1] > 3 else xyz)
        noise_prev = noise

    return out

def temporal_warp_sequence(points_seq, max_skip=2):
    seq_len = len(points_seq)
    max_skip = int(seq_len // 5)
    
    if (max_skip < 1):
        return points_seq, None
    
    warped = [frame.copy() for frame in points_seq]

    # Number of frames to remove
    n_skip = np.random.randint(1, max_skip + 1)

    # Random frame indices to replace (we keep the first and last one)
    skip_indices = np.random.choice(
        np.arange(1, seq_len-1),
        size=n_skip,
        replace=False
    )

    pad_frame = np.zeros_like(points_seq[0])

    for idx in skip_indices:
        warped[idx] = pad_frame.copy()

    return warped, skip_indices

def point_dropout_with_padding_sequence(points_seq, masks_seq, max_dropout_ratio=0.4):
    """
    Drop points by moving them to the padded point and updating the mask.
    Shape is preserved.
    """
    dropout_ratio = np.random.uniform(0.0, max_dropout_ratio)
    dropped_points_seq = []
    dropped_masks_seq = []

    for pc, mask in zip(points_seq, masks_seq):
        pc = pc.copy()
        mask = mask.copy()

        valid_idx = np.where(mask == 1)[0]
        if len(valid_idx) == 0:
            dropped_points_seq.append(pc)
            dropped_masks_seq.append(mask)
            continue

        num_drop = int(len(valid_idx) * dropout_ratio)
        if num_drop == 0:
            dropped_points_seq.append(pc)
            dropped_masks_seq.append(mask)
            continue
        
        if len(valid_idx) - num_drop < 3:
            dropped_points_seq.append(pc)
            dropped_masks_seq.append(mask)
            continue

        drop_idx = np.random.choice(valid_idx, num_drop, replace=False)

        # padded point (last index)
        pad_point = pc[-1, :3]

        pc[drop_idx, :3] = pad_point
        mask[drop_idx] = 0

        dropped_points_seq.append(pc)
        dropped_masks_seq.append(mask)

    return dropped_points_seq, dropped_masks_seq

def local_deformation_sequence(points_seq, scale=0.05):
    center = np.mean(points_seq[0][:, :3], axis=0)
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    deformed_seq = []
    for pc in points_seq:
        coords = pc[:, :3]
        dist = np.dot(coords - center, direction)
        warp = scale * np.sin(dist)
        coords = coords + warp[:, None] * direction
        pc_aug = np.hstack([coords, pc[:, 3:]]) if pc.shape[1] > 3 else coords
        deformed_seq.append(pc_aug)

    return deformed_seq


def temporal_dropout_with_padding(points_seq, masks_seq, drop_prob=0.3):
    """
    Drop entire frames by zeroing them out and setting masks to 0.
    Sequence length is preserved.
    """
    T = len(points_seq)
    keep = np.random.rand(T) > drop_prob

    # ensure at least 1–2 frames survive
    if keep.sum() < 2:
        keep[np.random.choice(T, 2, replace=False)] = True

    dropped_points = []
    dropped_masks = []

    for pc, mask, k in zip(points_seq, masks_seq, keep):
        if k:
            dropped_points.append(pc)
            dropped_masks.append(mask)
        else:
            dropped_points.append(np.zeros_like(pc))
            dropped_masks.append(np.zeros_like(mask))

    return dropped_points, dropped_masks


def random_point_dropout(points, max_ratio=0.3):
    keep = np.random.rand(len(points)) > np.random.uniform(0, max_ratio)

    # don't destroy the cloud
    if keep.sum() < 512:
        keep[np.random.choice(len(points), 512, replace=False)] = True

    return points[keep]