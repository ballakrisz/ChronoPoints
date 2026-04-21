import numpy as np

def rotate_sequence(points_seq, axis='z'):
    """
    Rotate a sequence of point clouds around a given axis.
    Handles extra per-point features (e.g., intensity) without modification.
    points_seq: list of np.ndarray [N, 3] or [N, F], F >= 3
    axis: 'x', 'y', or 'z'
    """
    theta = np.random.uniform(0, 2*np.pi) # consistent for the sequence
    if axis == 'z':
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    rotated_seq = []
    for pc in points_seq:
        coords = pc[:, :3]  # XYZ
        other_feats = pc[:, 3:] if pc.shape[1] > 3 else None
        coords = coords @ R.T
        if other_feats is not None:
            pc_aug = np.hstack([coords, other_feats])
        else:
            pc_aug = coords
        rotated_seq.append(pc_aug)
    return rotated_seq

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

def jitter_sequence(points_seq, sigma=0.01):
    """
    Apply random jittering to XYZ coordinates only. Other features remain unchanged.
    """
    noise = np.random.normal(0, sigma, points_seq[0][:, :3].shape) # consistent for the sequence
    
    jittered_seq = []
    for pc in points_seq:
        coords = pc[:, :3] + noise
        other_feats = pc[:, 3:] if pc.shape[1] > 3 else None
        if other_feats is not None:
            pc_aug = np.hstack([coords, other_feats])
        else:
            pc_aug = coords
        jittered_seq.append(pc_aug)
    return jittered_seq