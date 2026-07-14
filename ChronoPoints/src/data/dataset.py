from torch.utils.data import Dataset
import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
file = Path(__file__).resolve().parent

import json
from tqdm import tqdm
import copy
from typing import Dict, Tuple
from time import time
from multiprocessing import Pool, cpu_count
from data_utils import generate_unique_key, json_item_to_pcl_sequence, json_item_to_pcl_frame
from sampling import get_sampling_strategies, get_padding_strategies
from transform import rotate_sequence, jitter_sequence, point_dropout_with_padding_sequence, anisotropic_scale_sequence, temporal_warp_sequence
import torch
from collections import Counter
import random
from collections import defaultdict


def extract_movement_id(frame_path: str) -> str:
    # drop ".../frame_xxx.npz"
    return "/".join(frame_path.split("/")[:-1])

def load_and_return_wrapper(args):
    """
    Wrapper function to load a JSON item and convert it to a point cloud sequence.
    This function is used for multiprocessing to avoid pickling issues with the main dataset class.
    """
    json_item, synoff2cat, class_encoder, type_encoder, pad_token, data_root_dir, max_points_per_frame, frame_interval, sampling_strategy, padding_strategy = args

    key = generate_unique_key(json_item)
    dict_item = json_item_to_pcl_sequence(
        json_item=json_item,
        synoff2cat=synoff2cat,
        class_encoder=class_encoder,
        type_encoder=type_encoder,
        pad_token=pad_token,
        data_root_dir=data_root_dir,
        max_points_per_frame=max_points_per_frame,
        frame_interval=frame_interval,
        sampling_strategy=sampling_strategy,
        padding_strategy=padding_strategy,
        )
    return key, dict_item


def single_frame_load_and_return_wrapper(args, traj_eval=False):
    """
    Wrapper function to load a JSON item and convert it to a point cloud sequence.
    This function is used for multiprocessing to avoid pickling issues with the main dataset class.
    """
    file_path, synoff2cat, class_encoder, type_encoder, pad_token, data_root_dir, max_points_per_frame, frame_interval, sampling_strategy, padding_strategy = args

    key = generate_unique_key(file_path)
    dict_item = json_item_to_pcl_frame(
        file_path=file_path,
        synoff2cat=synoff2cat,
        class_encoder=class_encoder,
        type_encoder=type_encoder,
        pad_token=pad_token,
        data_root_dir=data_root_dir,
        max_points_per_frame=max_points_per_frame,
        frame_interval=frame_interval,
        sampling_strategy=sampling_strategy,
        padding_strategy=padding_strategy,
        traj_eval=traj_eval
    )
    return key, dict_item


def balance_by_label(data_dict, labels, seed=42):
    """
    Downsample majority classes so all classes have the same number of samples.

    Args:
        data_dict (Dict[str, Any]): key -> sample
        labels (List[int]): label per key (same order as data_dict.keys())
        seed (int): reproducibility

    Returns:
        new_data_dict, new_labels
    """
    random.seed(seed)

    keys = list(data_dict.keys())
    label_to_keys = defaultdict(list)

    for key, label in zip(keys, labels):
        label_to_keys[label].append(key)

    min_count = min(len(v) for v in label_to_keys.values())

    selected_keys = []
    for label, klist in label_to_keys.items():
        selected_keys.extend(random.sample(klist, min_count))

    random.shuffle(selected_keys)

    new_data_dict = {k: data_dict[k] for k in selected_keys}
    new_labels = [labels[keys.index(k)] for k in selected_keys]

    return new_data_dict, new_labels



class PointSeriesDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        split: str,
        max_points_per_frame: int = 100,
        single_return_only: bool = True,
        preload: bool = True,
        sampling_strategy: str = "inverse_density",
        padding_strategy: str = "zero_padding",
        sequence_format: str = "f10g1",
        seed = 0,
        balance = False
    ):
        """
        Dataset for temporally-ordered point cloud sequences.
        
        Args:
            data_root_dir (str): Root directory containing the dataset files.
            split (str): Dataset split to use.
                Options:
                    - 'train' : Training set
                    - 'val'   : Validation set
                    - 'test'  : Test set
            max_points_per_frame (int): Maximum number of points per frame after sampling.
            single_return_only (bool): If True, use only single return data; otherwise, use mixed return (single and double) data.
            preload (bool): If True, preload all data into memory; otherwise, load on-the-fly.
            sampling_strategy (str): Strategy for sampling point clouds.
                Options:
                    - 'farthest'         : Farthest Point Sampling
                    - 'random'           : Random Uniform Sampling
                    - 'inverse_density'  : Inverse Density Importance Sampling
                    - 'temporal'         : Temporal Consistency Sampling
            padding_strategy (str): Strategy for padding point clouds.
                Options:
                    - 'zero_padding'     : Zero Padding
            sequence_format (str): Format of the point cloud sequences.
                Options:
                    - 'f5g0'  : 5 frames, 0 gaps
                    - 'f10g1' : 10 frames, 1 gap
                    - 'f15g1' : 15 frames, 1 gap
                    - 'f20g1' : 20 frames, 1 gap
        """
        self._validate_args(data_root_dir, split, max_points_per_frame, preload, single_return_only, sampling_strategy, padding_strategy)

        self.pad_token = "<PAD/>"
        self.frame_interval = 100_000_000  # 100 ms

        self.data_root_dir = data_root_dir
        self.split = split
        self.max_points_per_frame = max_points_per_frame
        self.preload = preload
        self.sampling_strategy = sampling_strategy
        self.padding_strategy = padding_strategy
        self.sequence_format = sequence_format
        self.augment = split == "train"

        # Load dictionaries for class encoding
        self.synoff2cat = json.loads(Path(os.path.join(data_root_dir, "synsetoffset2category.json")).read_text())
        self.cat2synoff = {v: k for k, v in self.synoff2cat.items()}
        self.type_encoder = json.loads(Path(os.path.join(data_root_dir, "type_encoder.json")).read_text())
        self.class_encoder = json.loads(Path(os.path.join(data_root_dir, "class_encoder.json")).read_text())
        
        # Determine available classes/types by encoder index (>= 0)
        self.available_classes = [
            name for name, idx in self.class_encoder.items()
            if idx >= 0
        ]
        self.num_classes = len(self.available_classes)

        self.available_types = [
            name for name, idx in self.type_encoder.items()
            if idx >= 0
        ]
        self.num_types = len(self.available_types)

        # Set up JSON paths for the data splits 
        self.load_json_splits(single_return_only)

        # Main data container: maps ID -> point cloud, classs, timestamps, etc.
        self.data_dict: Dict[str, Dict] = {}
        self.labels = []
        self.movement_ids = []
        self.load_data(split)
        
        if split == "train" and balance:
            self.balance_types(seed=seed)

        print(f"Split '{split}' loaded with available classes: {self.available_classes} and types: {self.available_types}")
        print(f'The {split} dataset contains {len(self.data_dict)} point cloud sequences')
        


    def _validate_args(self, data_root_dir, split, max_points_per_frame, preload, single_return_only, sampling_strategy, padding_strategy):
            # Validate enums
            if split not in ['train', 'val', 'test']:
                raise ValueError(f"Invalid split: {split}. Must be one of {['train', 'val', 'test']}")
            if sampling_strategy not in get_sampling_strategies():
                raise ValueError(f"Invalid sampling strategy: {sampling_strategy}. Must be one of {get_sampling_strategies()}")
            if padding_strategy not in get_padding_strategies():
                raise ValueError(f"Invalid padding strategy: {padding_strategy}. Must be one of {get_padding_strategies()}")

            # Validate types
            if not isinstance(max_points_per_frame, int) or max_points_per_frame <= 0:
                raise ValueError("max_points_per_frame must be a positive integer")
            if not isinstance(preload, bool):
                raise TypeError("preload must be a boolean")
            if not isinstance(single_return_only, bool):
                raise TypeError("single_return_only must be a boolean")

            # Validate path
            data_root = Path(data_root_dir)
            if not data_root.is_dir():
                raise ValueError(f"data_root_dir must be a valid directory path, got: {data_root_dir}")
            

    def balance_types(self, seed=42):
        """
        Balance dataset by object_type via random downsampling.
        """
        print("Balancing PointSeriesDataset by object_type...")

        old_len = len(self.data_dict)

        self.data_dict, self.labels = balance_by_label(
            self.data_dict,
            self.labels,
            seed=seed
        )

        # Keep movement_ids aligned
        key_to_movement = dict(zip(
            list(self.data_dict.keys()),
            self.movement_ids
        ))
        self.movement_ids = [key_to_movement[k] for k in self.data_dict.keys()]

        print(f"Dataset balanced: {old_len} → {len(self.data_dict)} samples")
        counts = Counter(self.labels)  # PointSeriesDataset
        print("Balanced type distribution:", counts)


    def load_json_splits(self, single_return_only: bool) -> None:
        """
        Setup train/val/test JSON paths
        
        Args:
            single_return_only (bool): If True, use single return data; otherwise, use mixed return (single and double) data.
        """
        json_prefix = "single_return" if single_return_only else "mixed_return"
        self.train_split = os.path.join(
            self.data_root_dir,
            "train_test_split_{}_no_overlap/train.json".format(
                self.sequence_format, json_prefix)
        )

        self.val_split = os.path.join(
            self.data_root_dir,
            "train_test_split_{}_no_overlap/val.json".format(
                self.sequence_format, json_prefix)
        )

        self.test_split = os.path.join(
            self.data_root_dir,
            "train_test_split_{}_no_overlap/test.json".format(
                self.sequence_format, json_prefix)
        )

        self.split_map = {
            "train": self.train_split,
            "val": self.val_split,
            "test": self.test_split
        }

    def query_sample_distribution(self) -> None:
        """
        Create and print sample distribution by object class and object type in the dataset 
        """
        print(f"\nDataset sample distribution for split {self.split}:")

        # ------------------ Count types ------------------
        type_counts = Counter()
        for sample in self:
            _, _, _, _, type_label = sample
            type_counts[type_label] += 1

        print("\n=========== Type Distribution ===========")
        for type_name, type_id in self.type_encoder.items():
            if type_name == self.pad_token:
                continue  # skip <PAD/>
            count = type_counts.get(type_id, 0)
            print(f"Type ID: {type_id:<2} ({type_name:<12}) - Count: {count}")

        # ------------------ Count classes ------------------
        class_counts = Counter()
        for sample in self:
            _, _, _, class_label, _ = sample
            class_counts[class_label] += 1

        print("\n=========== Class Distribution ===========")
        for class_name, class_id in self.class_encoder.items():
            if class_name == self.pad_token:
                continue  # skip <PAD/>
            count = class_counts.get(class_id, 0)
            print(f"Class ID: {class_id:<2} ({class_name:<8}) - Count: {count}")


    def __len__(self) -> int:
        return len(self.data_dict)


    def _get_shared_context(self):
        """Returns the static context needed to convert json_items to pcl sequences"""
        return (
            self.synoff2cat,
            self.class_encoder,
            self.type_encoder,
            self.pad_token,
            self.data_root_dir,
            self.max_points_per_frame,
            self.frame_interval,
            self.sampling_strategy,
            self.padding_strategy,
        )
        
    


    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = list(self.data_dict.keys())[idx]

        if not self.preload:
            json_item = self.data_dict[key]['paths']
            dict_item = json_item_to_pcl_sequence(
                json_item,
                *self._get_shared_context()
            )
        else:
            dict_item = self.data_dict[key]
            
        # Dataset dict item structure:
        #   - 'point_clouds': List[np.ndarray]  # Sequence of point clouds
        #   - 'masks': List[np.ndarray]         # Valid point masks for each point cloud
        #   - 'velocities': List[np.ndarray]    # Per-frame velocities (x,y,z) for each point cloud in [m/s]
        #   - 'timestamps': List[np.ndarray]    # Timestamps for each frame
        #   - 'object_class': int               # Encoded object class
        #   - 'object_type': int                # Encoded object type

        """
        Augmentation:
        - Perform augmentation inside __getitem__ rather than in the training loop to reduce data loading time.
        - With DataLoader(num_workers>0), each worker calls __getitem__ in parallel, allowing parallel augmentation.
        - If augmentation is done in the training loop, it happens sequentially, which is slower.
        - DataLoader prefetches batches in the background using worker processes while the main process trains the model on previous batches.
        - Ensure different random seeds per worker for proper randomness in augmentations:
            e.g., worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id) --> yeah, well this does not work... you need to seed them based on the global RNG state. 
        """        
        skip_indices = None
        point_seq_np = dict_item['point_clouds']
        if self.augment:
            if np.random.rand() < 0.7:
                point_seq_np = rotate_sequence(point_seq_np)

            if np.random.rand() < 0.5:
                point_seq_np = anisotropic_scale_sequence(point_seq_np)

            if np.random.rand() < 0.5:
                point_seq_np = jitter_sequence(point_seq_np)

            if np.random.rand() < 0.3:
                point_seq_np, skip_indices = temporal_warp_sequence(point_seq_np)


                
                
        # old:
        # if self.augment: 
        #     # Apply augmentations consistently across the sequence 
        #     point_seq_np = rotate_sequence(point_seq_np) 
        #     point_seq_np = shift_sequence(point_seq_np) 
        #     point_seq_np = scale_sequence(point_seq_np) 
        #     point_seq_np = jitter_sequence(point_seq_np)

        """
        Return type:
        - Return tensors instead of NumPy arrays to avoid conversion overhead in the training loop (similar to augmentation).
        - Converting NumPy arrays to tensors in the training loop is sequential and slows down training, while the workers are processing data in parallel.
        - Setting pin_memory=True in DataLoader allows tensors to be pinned in CPU memory, enabling faster CPU→GPU transfer.
        - If tensors are not pinned, the OS *could* swap out the memory pages (where they are stored) during transfer, forcing the GPU to copy them in a blocking manner, so it won't happen.
        - If they are pinned, the memory is page-locked, so the GPU can perform the transfer non-blockingly, as the OS can't swap the pages out.
            In this case, we can use non_blocking=True when moving data to GPU:
                batch_point_clouds = batch_point_clouds.to(device, non_blocking=True)
            * Normally, CPU waits for transfer to complete (blocking).
            * With non_blocking=True, CPU can continue preparing the next batch while GPU transfer occurs asynchronously.
        """
        point_seq_torch = torch.stack([torch.from_numpy(pc).float() for pc in point_seq_np], dim=0) # (T, N, 3)
        point_seq_torch = point_seq_torch[:, :, :3]  # Keep only XYZ for model input
        
        masks = torch.stack([torch.from_numpy(mask).bool() for mask in dict_item['masks']], dim=0) # (T, N)
        if skip_indices is not None:
            masks[skip_indices] = False
        
        velocities = torch.stack([torch.from_numpy(vel).float() for vel in dict_item['velocities']], dim=0)  # (T, 3)
        
        timestamps = dict_item['timestamps'] - dict_item['timestamps'][0]  # normalize to start from 0
        timestamps = torch.tensor(timestamps, dtype=torch.float32) / 1e6  # convert to milliseconds (T,)
        object_class = torch.tensor(dict_item['object_class'], dtype=torch.long) # (1,)
        object_type = torch.tensor(dict_item['object_type'], dtype=torch.long) # (1,)

        return (
            point_seq_torch,
            masks,
            velocities,
            timestamps,
            object_class,
            object_type,
        )


    def load_data(self, split: str) -> None:
        """
        Load split files and fill the dataset dictionary.
        
        Args:
            split (str): The dataset split to load ('train', 'val', 'test').
        Raises:
            ValueError: If the split name is invalid.
        """
        try:
            split_path = self.split_map[split]
        except KeyError:
            raise ValueError(f"Invalid split name: {split}. Expected one of {list(self.split_map.keys())}")

        with open(split_path, 'r') as file:
            items = json.load(file)

            if not self.preload:
                print(f"Loading paths from {split_path.split('/')[-1]}...")
                for json_item in tqdm(items):
                    self.data_dict[generate_unique_key(json_item)] = {
                        "paths": json_item,
                    }
            else:
                print(f"Preloading data from {split_path.split('/')[-1]}...")
                shared_context = self._get_shared_context()
                context = [(json_item, *shared_context) for json_item in items]
                # Use multiprocessing to load data in parallel
                with Pool(processes=cpu_count()) as pool:
                    for key, data in tqdm(pool.imap(load_and_return_wrapper, context), total=len(items)):
                        if data:
                            self.data_dict[key] = data

                            # ---- NEW PART ----
                            first_frame = data["paths"][0] if "paths" in data else items[len(self.labels)][0]
                            movement_id = extract_movement_id(first_frame)

                            self.movement_ids.append(movement_id)
                            self.labels.append(data["object_type"])






class SingleFrameDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        split: str,
        max_points_per_frame: int = 100,
        single_return_only: bool = True,
        preload: bool = True,
        sampling_strategy: str = "inverse_density",
        padding_strategy: str = "zero_padding",
        statistics: bool = False,
        traj_eval: bool = False,
        sequence_format: str = "f10g1",
        bbox_mode: int = 0,
        seed = 0,
        balance = False
    ):
        # self._validate_args(data_root_dir, split, max_points_per_frame, preload, single_return_only, sampling_strategy, padding_strategy)

        self.pad_token = "<PAD/>"
        self.frame_interval = 100_000_000  # 100 ms

        self.bbox_mode = bbox_mode
        self.statistics = statistics
        self.data_root_dir = data_root_dir
        self.max_points_per_frame = max_points_per_frame
        self.preload = preload
        self.sampling_strategy = sampling_strategy
        self.padding_strategy = padding_strategy
        self.traj_eval = traj_eval
        self.sequence_format = sequence_format

        # Load dictionaries for class encoding
        self.synoff2cat = json.loads(Path(os.path.join(data_root_dir, "synsetoffset2category.json")).read_text())
        self.cat2synoff = {v: k for k, v in self.synoff2cat.items()}
        self.type_encoder = json.loads(Path(os.path.join(data_root_dir, "type_encoder.json")).read_text())
        self.class_encoder = json.loads(Path(os.path.join(data_root_dir, "class_encoder.json")).read_text())
        
        # Determine available classes/types by encoder index (>= 0)
        self.available_classes = [
            name for name, idx in self.class_encoder.items()
            if idx >= 0
        ]
        self.num_classes = len(self.available_classes)

        self.available_types = [
            name for name, idx in self.type_encoder.items()
            if idx >= 0
        ]
        self.num_types = len(self.available_types)

        # Determine which split files to load
        self.load_json_splits(single_return_only)

        # Main data container: maps ID -> point cloud, classs, timestamps, etc.
        self.data_dict: Dict[str, Dict] = {}
        self.load_data(split)
        
        if split == "train" and balance:
            self.balance_types(seed=seed)
        
        print(f'Our {split} dataset contains {len(self.data_dict)} point clouds')
        print(f"Split '{split}' loaded with available classes: {self.available_classes} and types: {self.available_types}")


    def _validate_args(self, data_root_dir, split, max_points_per_frame, preload, single_return_only, sampling_strategy, padding_strategy):
            # Validate enums
            if split not in ['train', 'val', 'test']:
                raise ValueError(f"Invalid split: {split}. Must be one of {['train', 'val', 'test']}")
            if sampling_strategy not in get_sampling_strategies():
                raise ValueError(f"Invalid sampling strategy: {sampling_strategy}. Must be one of {get_sampling_strategies()}")
            if padding_strategy not in get_padding_strategies():
                raise ValueError(f"Invalid padding strategy: {padding_strategy}. Must be one of {get_padding_strategies()}")

            # Validate types
            if not isinstance(max_points_per_frame, int) or max_points_per_frame < 0:
                raise ValueError("max_points_per_frame must be a positive integer")
            if not isinstance(preload, bool):
                raise TypeError("preload must be a boolean")
            if not isinstance(single_return_only, bool):
                raise TypeError("single_return_only must be a boolean")

            # Validate path
            data_root = Path(data_root_dir)
            if not data_root.is_dir():
                raise ValueError(f"data_root_dir must be a valid directory path, got: {data_root_dir}")
            
            jsons = list(data_root.glob("*.json"))
            if len(jsons) != 3:
                raise ValueError(f"data_root_dir should contain exactly 3 JSON files, found: {[f.name for f in jsons]}")


    def balance_types(self, seed=42):
        """
        Balance dataset by object_type via random downsampling.
        """
        print("Balancing SingleFrameDataset by object_type...")

        random.seed(seed)

        label_to_keys = defaultdict(list)
        for key, data in self.data_dict.items():
            label_to_keys[int(data["object_type"])].append(key)

        min_count = min(len(v) for v in label_to_keys.values())

        selected_keys = []
        for label, keys in label_to_keys.items():
            selected_keys.extend(random.sample(keys, min_count))

        random.shuffle(selected_keys)

        old_len = len(self.data_dict)
        self.data_dict = {k: self.data_dict[k] for k in selected_keys}

        print(f"Dataset balanced: {old_len} → {len(self.data_dict)} samples")
        counts = Counter(int(d["object_type"]) for d in self.data_dict.values())
        print("Balanced type distribution:", counts)


    def load_json_splits(self, single_return_only: bool) -> None:
        """
        Setup train/val/test JSON paths
        
        Args:
            single_return_only (bool): If True, use single return data; otherwise, use mixed return (single and double) data.
        """
        json_prefix = "single_return" if single_return_only else "mixed_return"
        self.train_split = os.path.join(self.data_root_dir, f"train_test_split_{self.sequence_format}_no_overlap/train.json")
        self.val_split = os.path.join(self.data_root_dir, f"train_test_split_{self.sequence_format}_no_overlap/val.json")
        self.test_split = os.path.join(self.data_root_dir, f"train_test_split_{self.sequence_format}_no_overlap/test.json")
        self.split_map = {
            "train": self.train_split,
            "val": self.val_split,
            "test": self.test_split
        }


    def __len__(self) -> int:
        return len(self.data_dict)

    # normalize the data, so it has a mean of 0, and is within the range [-1, 1] (unit circle)
    def normalize(self,pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = list(self.data_dict.keys())[idx]
        data = self.data_dict[key]

        if not self.traj_eval:
            pcl = data['point_cloud']
            orig_pcl = copy.deepcopy(pcl)
            
            valid_indices = (orig_pcl != 0).any(axis=1)
            valid_points = torch.tensor(orig_pcl[valid_indices])
            min_coords = valid_points.min(dim=0).values
            max_coords = valid_points.max(dim=0).values

            # Step 2: Compute the bounding box diagonal vector
            diagonal_vector = max_coords - min_coords

            bbox_dims = max_coords[:3] - min_coords[:3]

            # Step 3: Compute the length of the diagonal (Euclidean norm)
            diagonal_length = torch.norm(diagonal_vector[:3])

            if self.bbox_mode == 1 or self.bbox_mode == 3:
                bbox_ret = bbox_dims
            elif self.bbox_mode == 2:
                bbox_ret = diagonal_length
            elif self.bbox_mode == 0:
                bbox_ret = diagonal_length
            else:
                raise ValueError("Invalid bbox_mode")

            pcl[:,:3] = self.normalize(pcl[:,:3])
            pcl[:,3] /= 255.0  # Normalize intensity to [0, 1]
            type_target = data['object_type']
            class_target = data['object_class']

            return pcl, type_target, orig_pcl, bbox_ret
        
        else:
            for i,pcl in enumerate(data['point_cloud']):
                if data["object_type"][i] == -1:
                    continue
                pcl[:,:3] = self.normalize(pcl[:,:3])
                pcl[:,3] /= 255.0  # Normalize intensity to [0, 1]
            type_target = data['object_type']
            class_target = data['object_class']
            return data['point_cloud'], type_target

    def _get_shared_context(self):
        """Returns the static context needed to convert json_items to pcl sequences"""
        return (
            self.synoff2cat,
            self.class_encoder,
            self.type_encoder,
            self.pad_token,
            self.data_root_dir,
            self.max_points_per_frame,
            self.frame_interval,
            self.sampling_strategy,
            self.padding_strategy,
        )

    def load_data(self, split: str) -> None:
        try:
            split_path = self.split_map[split]
        except KeyError:
            raise ValueError(f"Invalid split name: {split}. Expected one of {list(self.split_map.keys())}")

        with open(split_path, 'r') as file:
            
            if not self.traj_eval:
                items = json.load(file)
                flat = [p for sub in items for p in sub]
                flat = [p for p in flat if p != "<PAD/>"]
                valid_paths = list(set(flat))
                for file in tqdm(valid_paths, desc=f"Loading {split} data", total=len(valid_paths)):
                    shared_context = self._get_shared_context()
                    context = (file, *shared_context)
                    key, data = single_frame_load_and_return_wrapper(context, traj_eval=self.traj_eval)
                    if data:
                        self.data_dict[key] = data
                        
            else:
                items = json.load(file)
                for trajectory in tqdm(items, desc=f"Loading {split} data", total=len(items)):
                    shared_context = self._get_shared_context()
                    context = (trajectory, *shared_context)
                    key, data = single_frame_load_and_return_wrapper(context, traj_eval=self.traj_eval)
                    if data:
                        self.data_dict[key] = data

# dataset = SingleFrameDataset(
#     data_root_dir="/home/appuser/chrono_points_cls_benchmark",
#     split="test",
#     max_points_per_frame=64,
#     single_return_only=False,
#     preload=False,
#     sampling_strategy="farthest",
#     padding_strategy="zero_padding"
# )

# tmp = dataset[0]
# s=0

# pre_s = time()
# test_dataset_preload = PointSeriesDataset(
#     data_root_dir="/home/appuser/chrono_points_cls_benchmark",
#     split="test",
#     max_points_per_frame=64,
#     single_return_only=False,
#     preload=True,
#     sampling_strategy="inverse_density",
#     padding_strategy="zero_padding"
# )
# print(f"Preloaded dataset load time: {((time() - pre_s))} seconds")

# num_samples = len(test_dataset_preload)

# pre_s = time()
# for i in range(num_samples):
#     point_clouds, masks, timestamps, object_labels, object_types = test_dataset_preload[i]
# pre_elapsed = time() - pre_s
# print(f"Preloaded dataset avg access time: {(pre_elapsed)*10**3 / num_samples} milliseconds")


# print("\n")

# no_pre_s = time()
# test_dataset_no_preload = PointSeriesDataset(
#     data_root_dir="/home/appuser/chrono_points_cls_benchmark",
#     split="test",
#     max_points_per_frame=64,
#     single_return_only=False,
#     preload=False,
#     sampling_strategy="inverse_density",
#     padding_strategy="zero_padding"
# )
# print(f"Non-preloaded dataset load time: {((time() - no_pre_s))} seconds")

# no_pre_s = time()
# for i in range(num_samples):
#     point_clouds, masks, timestamps, object_labels, object_types = test_dataset_no_preload[i]
# no_pre_elapsed = time() - no_pre_s
# print(f"Non-preloaded dataset avg access time: {(no_pre_elapsed)*10**3 / num_samples} milliseconds")

# print("\n")

# print(f"Time saved in EACH epoch by preloading: {no_pre_elapsed - pre_elapsed} seconds")
