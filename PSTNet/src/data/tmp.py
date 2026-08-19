from torch.utils.data import Dataset
import os
import sys
import numpy as np
import hashlib
from pathlib import Path
import torch
import hashlib

sys.path.append(str(Path(__file__).resolve().parent))
file = Path(__file__).resolve().parent
sys.path.append(str(Path(__file__).resolve().parents[2])) # Add the src directory to sys.path


from dataset import PointSeriesDataset

from LIFT_benchmark.utils.classification_dataset import PointSeriesDataset2

split = "train"

dataset_old = PointSeriesDataset(
        data_root_dir="/home/appuser/LIFT_benchmark",
        split=split,
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format="f10g1"
    )

dataset_new = PointSeriesDataset2(
        data_root_dir="/home/appuser/LIFT_benchmark",
        split=split,
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sequence_format="f10g1"
    )


def tensor_hash(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

assert len(dataset_old) == len(dataset_new), (
    f"Length mismatch: {len(dataset_old)} vs {len(dataset_new)}"
)

for i in range(len(dataset_old)):
    np.random.seed(42)                # reset for old
    old = dataset_old[i]
    np.random.seed(42)                # reset for new
    new = dataset_new[i]

    # Compare all fields (as before)
    for j, (a, b) in enumerate(zip(old, new)):
        if isinstance(a, torch.Tensor):
            if a.dtype.is_floating_point:
                equal = torch.allclose(a, b, atol=0.0, rtol=0.0)
            else:
                equal = torch.equal(a, b)
            if not equal:
                raise AssertionError(f"Sample {i}, field {j} differs")
        else:
            if a != b:
                raise AssertionError(f"Sample {i}, field {j}: {a} != {b}")

    # Now compare hashes (they will be identical)
    assert tensor_hash(old[0]) == tensor_hash(new[0]), f"Hash mismatch at sample {i}"

print(f"✓ All {len(dataset_old)} samples are identical.")

