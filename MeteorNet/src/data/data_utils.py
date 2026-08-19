import hashlib
import numpy as np
import os
from typing import List, Dict
from sampling import sample_and_pad, sample_and_pad_single_frame 
import torch
import matplotlib.pyplot as plt


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
    folder_to_class,
    class_encoder,
    type_encoder,
    pad_token: str,
    data_root_dir: str,
    max_points_per_frame: int,
    frame_interval: int,
    sampling_strategy: str = "inverse_density",
    padding_strategy: str = "zero_padding",
):
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

            object_class = folder_to_class[file_path.split("/")[0]]
            object_classes.append(class_encoder[object_class])
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
    folder_to_class,
    class_encoder,
    type_encoder,
    pad_token: str,
    data_root_dir: str,
    max_points_per_frame: int,
    frame_interval: int,
    sampling_strategy: str = "inverse_density",
    padding_strategy: str = "zero_padding",
    traj_eval: bool = False
):
    
    point_clouds, timestamps, object_classes, object_types = [], [], [], []
    
    if not traj_eval:


        data = np.load(os.path.join(data_root_dir, file_path))
        point_clouds.append(data['pcl'])
        timestamps.append(data['timestamp'])

        orig = torch.from_numpy(point_clouds[0])
        valid_points = (orig != 0).any(dim=1)
        xyz = orig[valid_points][:, :3]
        dist = torch.norm(xyz.mean(dim=0)).item() if xyz.shape[0] > 0 else 0

        object_class = folder_to_class[file_path.split("/")[0]]
        object_classes.append(class_encoder[object_class])
        object_types.append(type_encoder[str(data['type'])])
        data.close()

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

                object_class = folder_to_class[file.split("/")[0]]
                object_classes.append(class_encoder[object_class])
                object_types.append(type_encoder[str(data['type'])])
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
