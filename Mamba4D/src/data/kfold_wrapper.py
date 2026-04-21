import copy
import numpy as np
import torch
import random
from torch.utils.data import ConcatDataset, Subset, DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
import bisect
from collections import Counter, defaultdict


def print_pcl_count_per_movement(labels, movement_ids, type_decoder):
    # class -> movement -> pcl_count
    class_to_movements = defaultdict(lambda: defaultdict(int))

    for label, mov_id in zip(labels, movement_ids):
        class_to_movements[label][mov_id] += 1

    print("\n=== PCL count per movement (FULL DUMP) ===")
    for label in sorted(class_to_movements):
        print(f"\nClass {type_decoder[label]}:")
        movements = class_to_movements[label]

        for mov_id, pcl_count in sorted(movements.items()):
            print(f"  {mov_id:<70} : {pcl_count:<3} sequences")
            
            

def constrained_group_kfold(
    labels,
    movement_ids,
    n_splits,
    seed=42,
):
    """
    Returns a list of (train_idx, val_idx) tuples.
    Enforces:
      - group integrity (movement-level)
      - >=1 movement per class per fold (if possible)
    """

    rng = random.Random(seed)

    # ---- Step 1: collect unique movements ----
    movement_to_label = {}
    movement_to_indices = defaultdict(list)

    for idx, (lbl, mov) in enumerate(zip(labels, movement_ids)):
        movement_to_indices[mov].append(idx)
        movement_to_label[mov] = lbl

    # ---- Step 2: group movements by class ----
    class_to_movements = defaultdict(list)
    for mov, lbl in movement_to_label.items():
        class_to_movements[lbl].append(mov)

    # ---- Sanity check ----
    for lbl, movs in class_to_movements.items():
        if len(movs) < n_splits:
            raise ValueError(
                f"Class {lbl} has only {len(movs)} movements, "
                f"cannot split into {n_splits} folds with guarantees."
            )

    # ---- Step 3: initialize empty folds (movement sets) ----
    folds = [set() for _ in range(n_splits)]

    # ---- Step 4: hard constraint assignment ----
    for lbl, movs in class_to_movements.items():
        rng.shuffle(movs)
        for i in range(n_splits):
            folds[i].add(movs[i])

    # ---- Step 5: distribute remaining movements ----
    assigned = set().union(*folds)

    remaining = [
        mov for mov in movement_to_label.keys()
        if mov not in assigned
    ]

    rng.shuffle(remaining)

    def fold_size(fold):
        return sum(len(movement_to_indices[m]) for m in fold)

    for mov in remaining:
        # assign to smallest fold (by sequence count)
        smallest_fold = min(
            range(n_splits),
            key=lambda i: fold_size(folds[i])
        )
        folds[smallest_fold].add(mov)

    # ---- Step 6: build index splits ----
    splits = []
    all_movements = set(movement_to_label.keys())

    for i in range(n_splits):
        val_movs = folds[i]
        train_movs = all_movements - val_movs

        val_idx = [
            idx
            for mov in val_movs
            for idx in movement_to_indices[mov]
        ]
        train_idx = [
            idx
            for mov in train_movs
            for idx in movement_to_indices[mov]
        ]

        splits.append((np.array(train_idx), np.array(val_idx)))

    return splits

class AugmentView(Dataset):
    def __init__(self, concat_dataset, augment: bool):
        assert hasattr(concat_dataset, "datasets")
        assert hasattr(concat_dataset, "cumulative_sizes")

        self.concat = concat_dataset
        self.augment = augment

    def __len__(self):
        return len(self.concat)

    def _resolve_index(self, idx):
        """
        Resolve global idx -> (dataset_idx, sample_idx)
        """
        dataset_idx = bisect.bisect_right(
            self.concat.cumulative_sizes, idx
        )
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.concat.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitem__(self, idx):
        ds_idx, sample_idx = self._resolve_index(idx)
        ds = self.concat.datasets[ds_idx]

        # ds is a PointSeriesDataset
        old = ds.augment
        ds.augment = self.augment
        out = ds[sample_idx]
        ds.augment = old

        return out

class PointSeriesKfoldWrapper:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        n_splits=5,
        batch_size=32,
        num_workers=4,
        seed=42,
        pin_memory=True,
        persistent_workers=True,
    ):
        """
        K-fold wrapper for PointSeriesDataset using (train ∪ val).

        Args:
            train_dataset (PointSeriesDataset)
            val_dataset   (PointSeriesDataset)
            n_splits (int): number of folds
        """

        self.seed = seed
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self._seen_val_splits = set()
        self.type_decoder = {v: k for k,v in train_dataset.type_encoder.items()}

        # Merge train + val
        self.full_dataset = ConcatDataset([train_dataset, val_dataset])

        self.labels = []
        self.movement_ids = []

        for ds in [train_dataset, val_dataset]:
            for i in range(len(ds)):
                _, _, _, _, _, object_type = ds[i]
                self.labels.append(object_type.item())
                self.movement_ids.append(ds.movement_ids[i])

        # self.skf = StratifiedGroupKFold(
        #     n_splits=n_splits,
        #     shuffle=True,
        #     random_state=seed,
        # )
        
        print_pcl_count_per_movement(
            self.labels,
            self.movement_ids,
            self.type_decoder,
        )

        self.indices = np.arange(len(self.full_dataset))
        
        
    @staticmethod
    def _extract_labels(datasets):
        labels = []
        for ds in datasets:
            labels.extend(
                [v["object_type"] for v in ds.data_dict.values()]
            )
        return labels
    
    @staticmethod
    def _extract_movement_ids(datasets):
        movement_ids = []
        for ds in datasets:
            movement_ids.extend(ds.movement_ids)
        return movement_ids

    @staticmethod
    def _worker_init_fn(seed):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def _check_split_uniqueness(self, fold, val_idx):
        # frozenset so it’s hashable
        val_signature = frozenset(val_idx)

        if val_signature in self._seen_val_splits:
            raise RuntimeError(
                f"[KFold ERROR] Validation split in fold {fold} "
                f"is IDENTICAL to a previous fold!"
            )

        self._seen_val_splits.add(val_signature)
        
    def _print_box(self, text: str, pad: int = 6, char: str = "="):
        text = f"{char*3}{' ' * (pad-3)}{text}{' ' * (pad-3)}{char*3}"
        border = char * len(text)
        print(f"\n{border}")
        print(text)
        print(border)
            
    def _check_overlap(self, fold, train_idx, val_idx):
        train_set = set(train_idx)
        val_set   = set(val_idx)

        intersection = train_set.intersection(val_set)

        print(f"\n---------- FOLD {fold} ----------")
        print(f"Train size: {len(train_set)}")
        print(f"Val size:   {len(val_set)}")
        print(f"Overlap:    {len(intersection)}")

        assert len(intersection) == 0, \
            f"Data leakage detected in fold {fold}!"
            
    def _check_group_integrity(self, fold, train_idx, val_idx):
        train_groups = {self.movement_ids[i] for i in train_idx}
        val_groups   = {self.movement_ids[i] for i in val_idx}

        overlap = train_groups & val_groups
        assert len(overlap) == 0, (
            f"Movement leakage detected in fold {fold}: {list(overlap)[:3]}"
        )
            
    def _print_label_distribution(self, fold, train_idx, val_idx):
        train_labels = [self.labels[i] for i in train_idx]
        val_labels   = [self.labels[i] for i in val_idx]

        train_dist = Counter(train_labels)
        val_dist   = Counter(val_labels)

        print("\nLabel distribution:")
        print("  Train:")
        for k in sorted(train_dist):
            print(f"    Class {self.type_decoder[k]}: {train_dist[k]}")

        print("  Val:")
        for k in sorted(val_dist):
            print(f"    Class {self.type_decoder[k]}: {val_dist[k]}")
    

    def folds(self):
        self._print_box(
            f"Creating {self.n_splits} CONSTRAINED folds for Cross Validation"
        )

        splits = constrained_group_kfold(
            labels=self.labels,
            movement_ids=self.movement_ids,
            n_splits=self.n_splits,
            seed=self.seed,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            self._check_overlap(fold_idx, train_idx, val_idx)
            self._check_split_uniqueness(fold_idx, val_idx)
            self._print_label_distribution(fold_idx, train_idx, val_idx)

            yield fold_idx, *self._make_fold_loaders(train_idx, val_idx)


    def _make_fold_loaders(self, train_idx, val_idx):
            # wrap full dataset with augmentation control
            train_view = AugmentView(self.full_dataset, augment=True)
            val_view   = AugmentView(self.full_dataset, augment=False)

            train_loader = DataLoader(
                Subset(train_view, train_idx),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                worker_init_fn=self._worker_init_fn,
            )

            val_loader = DataLoader(
                Subset(val_view, val_idx),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
            )

            return train_loader, val_loader
