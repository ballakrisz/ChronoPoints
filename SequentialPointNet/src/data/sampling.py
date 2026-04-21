from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.neighbors import KDTree
from abc import ABC, abstractmethod
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PaddingStrategy(ABC):
    """
    Abstract base class for padding strategies.
    Each subclass must implement the `pad` method.
    """
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'name'):
            PaddingStrategy.registry[cls.name] = cls

    @abstractmethod
    def pad(self, point_sequence: List[np.ndarray], num_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Pads a sequence of point clouds to num_samples per frame.
        
        Returns:
            padded_points: List of np.ndarray with shape (num_samples, D)
            masks: List of np.ndarray with shape (num_samples,)
        """
        pass


class ZeroPadding(PaddingStrategy):
    name = "zero_padding"

    def pad(self, point_sequence: List[np.ndarray], num_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Zero-pad the points to ensure they have num_samples points.

        Args:
            point_sequence: List of np.ndarray, each of shape (N_i, D)
            num_samples: Desired number of samples

        Returns:
            Tuple of (padded_points, masks)
            padded_points: List of np.ndarray, each of shape (num_samples, D)
            masks: List of np.ndarray, each of shape (num_samples,), 1 for real points, 0 for padded
        """
        padded_points = []
        masks = []

        for points in point_sequence:
            N, F = points.shape
            if N >= num_samples:
                selected = points[:num_samples]
                mask = np.ones(num_samples, dtype=int)
            else:
                selected = np.zeros((num_samples, F), dtype=points.dtype)
                selected[:N] = points
                mask = np.concatenate([np.ones(N, dtype=int), np.zeros(num_samples - N, dtype=int)])

            padded_points.append(selected)
            masks.append(mask)

        return padded_points, masks

class BoundingBoxNoisePadding(PaddingStrategy):
    name = "bbox_noise_padding"

    def pad(self, point_sequence: List[np.ndarray], num_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Pads point clouds by sampling new points within a radius (10% of bounding box diagonal)
        around existing points.

        Args:
            point_sequence: List of np.ndarray, each of shape (N_i, D)
            num_samples: Desired number of samples per frame

        Returns:
            Tuple of (padded_points, masks)
            padded_points: List of np.ndarray, each of shape (num_samples, D)
            masks: List of np.ndarray, each of shape (num_samples,), 1 for real points, 0 for padded
        """
        padded_points = []
        masks = []

        for points in point_sequence:
            N, F = points.shape
            if N >= num_samples:
                selected = points[:num_samples]
                mask = np.ones(num_samples, dtype=int)
            else:
                # Compute bounding box
                min_xyz = points[:, :3].min(axis=0)
                max_xyz = points[:, :3].max(axis=0)
                bbox_diag = np.linalg.norm(max_xyz - min_xyz)
                radius = 0.01 * bbox_diag

                # Sample additional points near existing ones
                num_to_pad = num_samples - N
                pad_idxs = np.random.randint(0, N, size=num_to_pad)
                base_points = points[pad_idxs]
                
                # Create noise
                noise = np.random.normal(scale=radius, size=(num_to_pad, 3))
                
                padded = np.copy(base_points)
                padded[:, :3] += noise

                if F > 3:
                    padded[:, 3:] = 0  # Zero-fill additional feature dimensions if they exist

                selected = np.concatenate([points, padded], axis=0)
                mask = np.concatenate([np.ones(N, dtype=int), np.zeros(num_to_pad, dtype=int)])

                # fig = plt.figure(figsize=(12, 5))

                # # Original
                # ax1 = fig.add_subplot(121, projection='3d')
                # ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='blue')
                # ax1.set_title("Original Point Cloud")
                # ax1.set_xlabel("X")
                # ax1.set_ylabel("Y")
                # ax1.set_zlabel("Z")

                # # Padded
                # ax2 = fig.add_subplot(122, projection='3d')
                # ax2.scatter(selected[:, 0], selected[:, 1], selected[:, 2], s=5, c='green')
                # ax2.set_title("Padded to 512 Points")
                # ax2.set_xlabel("X")
                # ax2.set_ylabel("Y")
                # ax2.set_zlabel("Z")

                # plt.suptitle(f"Padding Visualization")
                # plt.tight_layout()
                # plt.show()

            padded_points.append(selected)
            masks.append(mask)

        return padded_points, masks

def get_padding(strategy_name: str) -> PaddingStrategy:
    """
    Returns a padding strategy based on the provided name.
    Args:
        strategy_name (str): Name of the padding strategy.
    Returns:
        PaddingStrategy: An instance of the specified padding strategy.
    """
    cls = PaddingStrategy.registry.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown padding strategy: {strategy_name}")
    return cls()


def get_padding_strategies() -> List[str]:
    """
    Returns a list of available padding strategy names.

    Returns:
        List[str]: List of names of available padding strategies.
    """
    return list(PaddingStrategy.registry.keys())


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.
    Each subclass must implement the `sample` method.
    """
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'name'):
            SamplingStrategy.registry[cls.name] = cls

    @abstractmethod
    def sample(self, point_sequence: List[np.ndarray], num_samples: int, seed: int = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Samples points from a sequence of point clouds respectively.
        
        Args:
            point_sequence: List of np.ndarray, each of shape (N_i, F)
            num_samples: Number of points to sample per frame
            seed: Optional int random seed for reproducibility
        Returns:
            sampled_points: List of np.ndarray, each of shape (num_samples, F)
        """
        pass


class FarthestPointSampling(SamplingStrategy):
    name = "farthest"

    def sample(self, point_sequence, num_samples, seed=None):
        """
        Perform farthest point sampling per frame with zero-padding and masks.

        Args:
            point_sequence: List of np.ndarray, each shape (N_i, F)
            num_samples: Number of points to sample per frame
            seed: Optional int random seed for reproducibility

        Returns:
            sampled_points: List of np.ndarray, each shape (num_samples, F)
        """
        if seed is not None:
            np.random.seed(seed)

        sampled_points = []

        for points in point_sequence:
            N, F = points.shape
            if N <= num_samples:
                selected = points
            else:
                sampled_indices = np.zeros(num_samples, dtype=int)
                sampled_indices[0] = np.random.randint(N)

                distances = np.full(N, np.inf)

                for i in range(1, num_samples):
                    current_point = points[sampled_indices[i - 1]]
                    dists = np.linalg.norm(points - current_point, axis=1)
                    distances = np.minimum(distances, dists)
                    sampled_indices[i] = np.argmax(distances)

                selected = points[sampled_indices]

            sampled_points.append(selected)

        return sampled_points  



class RandomPointSampling(SamplingStrategy):
    name = "random"

    def sample(self, point_sequence, num_samples, seed=None):
        """
        Randomly sample points per frame in a sequence, with zero-padding and masking.

        Args:
            point_sequence: List of np.ndarray, each of shape (N_i, F)
            num_samples: Number of points to sample per frame
            seed: Optional int for reproducibility

        Returns:
            sampled_points: List of np.ndarray, each of shape (num_samples, F)
        """
        if seed is not None:
            np.random.seed(seed)

        sampled_points = []

        for points in point_sequence:
            N = points.shape[0]
            if N <= num_samples:
                selected = points
            else:
                sampled_indices = np.random.choice(N, size=num_samples, replace=False)
                selected = points[sampled_indices]
            sampled_points.append(selected)

        return sampled_points



class InverseDensitySampling(SamplingStrategy):
    name = "inverse_density"

    def __init__(self, k=10):
        """
        Sample points in each frame using Inverse Density Importance Sampling (IDIS)

        Args:
            k: Number of neighbors used for local density estimation
        """
        self.k = k

    def sample(self, point_sequence, num_samples, seed=None):
        """
        Perform Inverse Density Importance Sampling (IDIS) with zero-padding and masking across a sequence of point clouds.

        Args:
            point_sequence: List of np.ndarray, each of shape (N_i, F)
            num_samples: Number of points to sample per frame
            seed: Optional random seed for reproducibility

        Returns:
            sampled_points: List of np.ndarray, each of shape (num_samples, F)
        """
        if seed is not None:
            np.random.seed(seed)

        sampled_points = []

        for points in point_sequence:
            N = points.shape[0]
            if N <= num_samples:
                sampled_points.append(points)
                continue

            # Estimate density
            nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(points)
            distances, _ = nbrs.kneighbors(points)
            avg_knn_distances = np.mean(distances[:, 1:], axis=1)
            weights = avg_knn_distances / np.sum(avg_knn_distances)

            sampled_indices = np.random.choice(N, size=num_samples, replace=False, p=weights)
            selected = points[sampled_indices]
            sampled_points.append(selected)

        return sampled_points



class TemporalConsistencySampling(SamplingStrategy):
    name = "temporal"

    def __init__(self, radius=0.05):
        """
        Sample points in each frame based on temporal consistency.

        Args:
            radius: Neighborhood radius for temporal consistency
        """
        self.radius = radius

    def sample(self, point_sequence, num_samples, seed=None):
        """
        Perform temporal point sampling with zero-padding and masks.
        If fewer than num_samples points, zero-pad and return mask.

        Args:
            point_sequence: List of np.ndarray, each (N_i, F)
            num_samples: Desired number of output points per frame
            seed: Optional random seed

        Returns:
           sampled_points: List of np.ndarray, each of shape (num_samples, F)
        """
        if seed is not None:
            np.random.seed(seed)

        trees = [KDTree(pc) for pc in point_sequence]
        sampled_points = []

        for i, points in enumerate(point_sequence):
            N = points.shape[0]
            scores = np.zeros(N, dtype=int)

            for j, tree in enumerate(trees):
                if i == j:
                    continue
                neighbors = tree.query_radius(points, r=self.radius)
                for idx_point, n in enumerate(neighbors):
                    if len(n) > 0:
                        scores[idx_point] += 1

            ranked_indices = np.argsort(-scores)

            if N <= num_samples:
                selected = points[ranked_indices]
            else:
                selected = points[ranked_indices[:num_samples]]

            sampled_points.append(selected)

        return sampled_points



def get_sampler(strategy_name: str) -> SamplingStrategy:
    """
    Returns a sampling strategy based on the provided name.

    Args:
        strategy_name (str): Name of the sampling strategy.
    Returns:
        SamplingStrategy: An instance of the specified sampling strategy.
    """
    cls = SamplingStrategy.registry.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}")
    if strategy_name == "inverse_density":
        return cls(k=10)
    if strategy_name == "temporal":
        return cls(radius=0.5)
    return cls()


def get_sampling_strategies() -> List[str]:
    """
    Returns a list of available sampling strategy names.
    
    Returns:
        List[str]: List of names of available sampling strategies.
    """
    return list(SamplingStrategy.registry.keys())


def normalize_sequence(pcl_seq_orig, pad_frames):
    """
    Normalize a sequence of point clouds that include (x, y, z, intensity).

    - xyz are normalized together (global unit sphere)
    - intensity is divided by 255
    """
    # Only consider non-padded frames for computing normalization stats (padded frames are handled after normalization, so that they don't affect the true distribution)
    pcl_seq = [np.asarray(pc) for pc, pad in zip(pcl_seq_orig, pad_frames) if pad == 0]

    # Concatenate all xyz for global normalization
    all_xyz = np.concatenate([pc[:, :3] for pc in pcl_seq], axis=0)
    centroid = np.mean(all_xyz, axis=0)
    all_xyz_centered = all_xyz - centroid
    m = np.max(np.sqrt(np.sum(all_xyz_centered ** 2, axis=1)))

    norm_seq = []
    for i, pc in enumerate(pcl_seq_orig):
        if pad_frames[i] == 1: # padded frame, keep as is
            norm_seq.append(pc)
            continue
        xyz = (pc[:, :3] - centroid) / (m + 1e-8)
        intensity = pc[:, 3:] / 255.0
        pc_norm = np.hstack([xyz, intensity])
        norm_seq.append(pc_norm)

    return norm_seq



def detect_and_visualize_true_outliers(pcl_sequence, pad_frames, json_item, factor=9.0, show=False, per_frame=False):
    """
    Detect true outliers in a sequence of point clouds and optionally visualize them.
    Outliers are points much farther from the centroid than the median distance.

    Parameters:
        pcl_seq : list[np.ndarray] of shape (T, N, 4)
        factor : float
            How many times the median distance a point must exceed to be considered an outlier
        show : bool
            Whether to display the 3D visualization

    Returns:
        outlier_masks : list[np.ndarray of bool] per frame
        outlier_counts : list[int] per frame
        outlier_ratios : list[float] per frame
    """
    # Only consider non-padded frames (padded frames are handled after normalization, so that they don't affect stats)
    pcl_seq = [np.asarray(pc) for i, pc in enumerate(pcl_sequence) if pad_frames[i] ==0]



    outlier_masks = []
    outlier_counts = []
    outlier_ratios = []

    has_outliers = False

    # per-frame outlier detection
    if per_frame:
        for pc in pcl_seq:
            xyz = pc[:, :3]
            centroid = np.mean(xyz, axis=0)
            dists = np.linalg.norm(xyz - centroid, axis=1)
            median_dist = np.median(dists)
            mask = dists > (factor * median_dist)
            outlier_masks.append(mask)
            outlier_counts.append(int(np.sum(mask)))
            outlier_ratios.append(float(np.sum(mask) / len(mask)))

            if np.any(mask):
                has_outliers = True
    else:
        # --- Compute global centroid and median distance ---
        all_xyz = np.concatenate([pc[:, :3] for pc in pcl_seq], axis=0)
        centroid = np.mean(all_xyz, axis=0)
        all_dists = np.linalg.norm(all_xyz - centroid, axis=1)
        median_dist = np.median(all_dists)

        # --- Global outlier detection - sequence level (uncomment to use) ---
        for pc in pcl_seq:
            xyz = pc[:, :3]
            dists_frame = np.linalg.norm(xyz - centroid, axis=1)
            mask = dists_frame > (factor * median_dist)
            outlier_masks.append(mask)
            outlier_counts.append(int(np.sum(mask)))
            outlier_ratios.append(float(np.sum(mask) / len(mask)))

            if np.any(mask):
                has_outliers = True

    if show and has_outliers:
        for i, (pc, mask) in enumerate(zip(pcl_seq, outlier_masks)):
                xyz = pc[:, :3]
                outlier_points = xyz[mask]
                if len(outlier_points) > 0:
                    print(f"Outlier(s) in frame {json_item[i]} at coords:  {outlier_points}")

    # --- Visualization ---
    if show and has_outliers:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap("tab10")

        for i, (pc, mask) in enumerate(zip(pcl_seq, outlier_masks)):
            xyz = pc[:, :3]
            # Normal points
            normal_points = xyz[~mask]
            ax.scatter(normal_points[:,0], normal_points[:,1], normal_points[:,2],
                       color=cmap(i % 10), s=20, label=f'Frame {i} normal')
            # Outlier points
            outlier_points = xyz[mask]
            if len(outlier_points) > 0:
                ax.scatter(outlier_points[:,0], outlier_points[:,1], outlier_points[:,2],
                           color='red', s=50, label=f'Frame {i} outlier')

        # Equal axis scaling
        all_coords = np.concatenate([pc[:, :3] for pc in pcl_seq], axis=0)
        x_mid, y_mid, z_mid = np.mean(all_coords, axis=0)
        max_range = np.max(np.ptp(all_coords, axis=0)) / 2
        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sequence True Outliers (red) - Factor {factor}')
        ax.legend()
        plt.show()

    # if there was padding, re-insert masks for padded frames (set to all False) --> this shenanigan is needed to keep alignment of mask length with original sequence length
    if len(pcl_sequence) != len(pcl_seq):
        full_outlier_masks = []
        pad_idx = 0
        for is_pad in pad_frames:
            if is_pad == 1:
                full_outlier_masks.append(np.array([False], dtype=bool))  # only 1 point in padded frame, and treat is as non-outlier
            else:
                full_outlier_masks.append(outlier_masks[pad_idx])
                pad_idx += 1
        outlier_masks = full_outlier_masks

    return outlier_masks, outlier_counts, outlier_ratios


def sample_and_pad_single_frame(
    point_sequence: List[np.ndarray],
    sampler: str,
    padding: str,
    num_samples: int,
    seed: int = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Applies sampling followed by padding.

    Args:
        point_sequence: List of np.ndarray, each of shape (N_i, D)
        sampler: Name of the sampling strategy to use
        padding: Name of the padding strategy to use
        num_samples: Number of points to sample and pad to per frame
        seed: Optional random seed for reproducibility

    Returns:
        padded_points: List[np.ndarray] of shape (num_samples, D)
        masks: List[np.ndarray] of shape (num_samples,)
    """
    if num_samples == 0:
        return point_sequence, np.ones((len(point_sequence),len(point_sequence[0])))
    sampler = get_sampler(sampler)
    padding = get_padding(padding)

    sampled_unpadded = sampler.sample(point_sequence, num_samples, seed)

    padded_points, masks = padding.pad(sampled_unpadded, num_samples)

    return padded_points, masks



def sample_and_pad(
    point_sequence: List[np.ndarray],
    sampler: str,
    padding: str,
    num_samples: int,
    seed: int = None,
    pad_frames: List[int] = None,
    json_item: List[str] = None,
    timestamps: List[float] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Applies sampling followed by padding.

    Args:
        point_sequence: List of np.ndarray, each of shape (N_i, D)
        sampler: Name of the sampling strategy to use
        padding: Name of the padding strategy to use
        num_samples: Number of points to sample and pad to per frame
        seed: Optional random seed for reproducibility

    Returns:
        padded_points: List[np.ndarray] of shape (num_samples, D)
        masks: List[np.ndarray] of shape (num_samples,)
    """
    if num_samples == 0:
        return point_sequence, np.ones((len(point_sequence),len(point_sequence[0])))
    sampler = get_sampler(sampler)
    padding = get_padding(padding)

    sampled_unpadded = sampler.sample(point_sequence, num_samples, seed)

    # Detect and visualize true outliers before normalization
    masks, counts, ratios = detect_and_visualize_true_outliers(sampled_unpadded, pad_frames=pad_frames, json_item=json_item)

    # Remove outliers from the sampled point clouds
    sampled_cleaned = []
    for pc, mask in zip(sampled_unpadded, masks):
        sampled_cleaned.append(pc[~mask])
        
    # Compute per-frame velocities (x,y,z only) before normalization
    velocities = []
    timestamps_s = np.array(timestamps).astype(np.float64) * 1e-9  # convert to seconds


    # Track last valid frame index
    last_valid_idx = None

    for t in range(len(sampled_cleaned)):
        if pad_frames is not None and pad_frames[t] == 1:
            # Padded frame → velocity zero
            velocities.append(np.zeros(3, dtype=np.float32))
        else:
            if last_valid_idx is None:
                # First non-padded frame → velocity zero
                velocities.append(np.zeros(3, dtype=np.float32))
            else:
                # Compute centroid displacement from last valid frame
                dt = timestamps_s[t] - timestamps_s[last_valid_idx]
                centroid_t = sampled_cleaned[t][:, :3].mean(axis=0)
                centroid_prev = sampled_cleaned[last_valid_idx][:, :3].mean(axis=0)
                v = (centroid_t - centroid_prev) / dt
                velocities.append(v)
            # Update last valid frame index
            last_valid_idx = t
            
        
    # Normalize the cleaned sampled sequence, preserving padded frames as is
    sampled_normalized = normalize_sequence(sampled_cleaned, pad_frames)

    padded_points, masks = padding.pad(sampled_normalized, num_samples)
    
    # if we had pad frames, set the corresponding masks to all zeros
    if 1 in pad_frames:
        for i, is_pad in enumerate(pad_frames):
            if is_pad == 1:
                masks[i] = np.zeros(num_samples, dtype=int)
    
    return padded_points, masks, velocities

