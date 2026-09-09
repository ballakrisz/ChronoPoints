import hashlib
import numpy as np
import os
from typing import List, Dict
from sampling import sample_and_pad, sample_and_pad_single_frame 
import torch
import matplotlib.pyplot as plt

# ======================================================================
# Visualization utility (for sequences, mainly for sanity check)
# ======================================================================
def visualize_sequence(pcl_sequence: List[np.ndarray], title: str = "Point Cloud Sequence"):
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for i, pc in enumerate(pcl_sequence):

        color = cmap(i % 10)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=20, c=color, alpha=1.0, label=f"Frame {i}")

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in this sense
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def visualize_sequences_multi_axis(sequences: List[List[np.ndarray]], 
                                   titles: List[str] = None, 
                                   main_title: str = "Point Cloud Sequences"):
    """
    Visualize multiple point cloud sequences in separate 3D subplots.
    
    sequences: list of sequences, where each sequence is a list of point clouds (frames)
               e.g., [original_seq, aug1_seq, aug2_seq, ...]
    titles: optional list of titles for each sequence
    main_title: main title of the figure
    """
    num_sequences = len(sequences)
    if titles is None:
        titles = [f"Sequence {i}" for i in range(num_sequences)]

    fig = plt.figure(figsize=(6 * num_sequences, 5))

    for seq_idx, pcl_sequence in enumerate(sequences):
        ax = fig.add_subplot(1, num_sequences, seq_idx + 1, projection="3d")
        cmap = plt.get_cmap("tab10")

        for frame_idx, pc in enumerate(pcl_sequence):
            color = cmap(frame_idx % 10)
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], 
                       s=20, c=color, alpha=0.8, label=f"Frame {frame_idx}")

        # Adjust plot to cubic bounds
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        ax.set_title(titles[seq_idx], fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(main_title, fontsize=18)
    plt.tight_layout()
    plt.show()

# ======================================================================
# Unique key generation for JSON items (just so we can store the entries in the dictionary)
# ======================================================================
def generate_unique_key(json_item: List[str]) -> str:
    joined_paths = "|".join(json_item)
    return hashlib.md5(joined_paths.encode()).hexdigest()


# ======================================================================
# Sequence-level point cloud loading and processing
# ======================================================================
def json_item_to_pcl_sequence(
    json_item: List[str],
    folder_to_class: Dict[str, str],
    class_encoder: Dict[str, int],
    type_encoder: Dict[str, int],
    pad_token: str,
    data_root_dir: str,
    max_points_per_frame: int,
    frame_interval: int,
    sampling_strategy: str = "inverse_density",
    padding_strategy: str = "zero_padding",
) -> Dict[str, np.ndarray]:
    point_clouds, timestamps, object_classes, object_types, pad_frames = [], [], [], [], []
    last_stamp = None
    curr_stamp = None


    for file_path in json_item:
        if file_path == pad_token:
            point_clouds.append(np.zeros((1, 4), dtype=np.float32))
            timestamps.append(last_stamp + frame_interval)
            object_classes.append(class_encoder[pad_token])
            object_types.append(type_encoder[pad_token])
            curr_stamp = last_stamp + frame_interval
            pad_frames.append(1)
        else:
            data = np.load(os.path.join(data_root_dir, file_path))
            point_clouds.append(data['pcl'])
            timestamps.append(data['timestamp'])
            
            folder_name = file_path.split("/")[0]
            class_name = folder_to_class.get(folder_name, folder_name.capitalize())
            object_classes.append(class_encoder[class_name])
            object_types.append(type_encoder[str(data['type'])])
            pad_frames.append(0)
            curr_stamp = data['timestamp']
            data.close()

        last_stamp = curr_stamp

    padded_point_clouds, masks, velocities = sample_and_pad(
        point_sequence=point_clouds,
        sampler=sampling_strategy,
        padding=padding_strategy,
        num_samples=max_points_per_frame,
        seed=42,
        pad_frames=pad_frames,
        json_item=json_item,
        timestamps=timestamps
    )

    object_class = set(object_classes) - {class_encoder[pad_token]}
    object_type = set(object_types) - {type_encoder[pad_token]}

    class_count = len(object_class)
    type_count = len(object_type)
    if class_count > 1 or type_count > 1:
        raise ValueError("Inconsistent object classes or types in the sequence.")

    seq_type = object_type.pop() 
    seq_class = object_class.pop()

    # Exclude Seagulls for now 
    if seq_type == type_encoder["Seagull"]:
        return None
        
    return {
        'point_clouds': np.array(padded_point_clouds, dtype=np.float32),
        'masks': np.array(masks, dtype=np.int64),
        'velocities': np.array(velocities, dtype=np.float32),
        'timestamps': np.array(timestamps, dtype=np.int64),
        'object_class': seq_class,
        'object_type': seq_type,
    }


# ======================================================================
# Frame-level point cloud loading and processing
# ======================================================================
def json_item_to_pcl_frame(
    file_path: str,
    synoff2cat: Dict[str, str],
    class_encoder: Dict[str, int],
    type_encoder: Dict[str, int],
    pad_token: str,
    data_root_dir: str,
    max_points_per_frame: int,
    frame_interval: int,
    sampling_strategy: str = "inverse_density",
    padding_strategy: str = "zero_padding",
    traj_eval: bool = False
) -> Dict[str, np.ndarray]:
    
    point_clouds, timestamps, object_classes, object_types = [], [], [], []
    
    if not traj_eval:


        data = np.load(os.path.join(data_root_dir, file_path))
        point_clouds.append(data['pcl'])
        timestamps.append(data['timestamp'])

        orig = torch.from_numpy(point_clouds[0])
        valid_points = (orig != 0).any(dim=1)
        xyz = orig[valid_points][:, :3]
        dist = torch.norm(xyz.mean(dim=0)).item() if xyz.shape[0] > 0 else 0

        object_class = synoff2cat[file_path.split("/")[0]]
        object_classes.append(class_encoder[object_class])
        object_types.append(type_encoder[str(data['type'])])
        data.close()
        
        # exlude seagulss
        if object_types[0] == type_encoder["Seagull"]:
            return None

        # if point_clouds[0].shape[0] > 10:
        #     return None

        padded_point_clouds, masks = sample_and_pad_single_frame(
            point_sequence=point_clouds,
            sampler=sampling_strategy,
            padding=padding_strategy,
            num_samples=max_points_per_frame,
            seed=42
        )
        
        return {
            'point_cloud': np.array(padded_point_clouds[0], dtype=np.float32),
            'mask': np.array(masks[0], dtype=np.int64),
            'timestamp': np.array(timestamps[0], dtype=np.int64),
            'object_class': np.array(object_classes[0], dtype=np.int64),
            'object_type': np.array(object_types[0], dtype=np.int64),
        }
        
    else:
        for file in file_path:
            if file == pad_token:
                point_clouds.append(np.zeros((1, 4), dtype=np.float32))
                timestamps.append(-1)
                object_classes.append(class_encoder[pad_token])
                object_types.append(type_encoder[pad_token])
            else:
                data = np.load(os.path.join(data_root_dir, file))
                point_clouds.append(data['pcl'])
                timestamps.append(data['timestamp'])

                orig = torch.from_numpy(point_clouds[0])
                valid_points = (orig != 0).any(dim=1)
                xyz = orig[valid_points][:, :3]
                dist = torch.norm(xyz.mean(dim=0)).item() if xyz.shape[0] > 0 else 0

                object_class = synoff2cat[file.split("/")[0]]
                object_classes.append(class_encoder[object_class])
                object_types.append(type_encoder[str(data['type'])])
                
                # if seaguls, skip entire sequence
                if object_types[-1] == type_encoder["Seagull"]:
                    return None
                data.close()


        padded_point_clouds, masks = sample_and_pad_single_frame(
            point_sequence=point_clouds,
            sampler=sampling_strategy,
            padding=padding_strategy,
            num_samples=max_points_per_frame,
            seed=42
        )
        return {
            'point_cloud': np.array(padded_point_clouds, dtype=np.float32),
            'mask': np.array(masks, dtype=np.int64),
            'timestamp': np.array(timestamps, dtype=np.int64),
            'object_class': np.array(object_classes, dtype=np.int64),
            'object_type': np.array(object_types, dtype=np.int64),
        }
