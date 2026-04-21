from dataset import SingleFrameDataset
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_combined_stats(datasets):
    stats = defaultdict(lambda: {
        "count": 0,
        "reflections": [],
        "bbox_diam": []
    })

    for dataset in datasets:
        for i in tqdm(range(len(dataset))):
            pcl, type_target, orig_pcl, bbox_ret = dataset[i]

            # Count valid reflections (ignore zero padding)
            valid_reflections = (orig_pcl[:, :3] != 0).any(axis=1).sum().item()

            # Handle bbox output
            if isinstance(bbox_ret, torch.Tensor) and bbox_ret.numel() == 3:
                bbox_diam = torch.norm(bbox_ret).item()
            else:
                bbox_diam = float(bbox_ret)

            class_id = int(type_target)

            stats[class_id]["count"] += 1
            stats[class_id]["reflections"].append(valid_reflections)
            stats[class_id]["bbox_diam"].append(bbox_diam)

    return stats

def print_stats_table(stats, class_names=None):
    print(f"{'Type':<10} {'Samples':<10} {'Reflections (mean±std)':<25} {'Avg bbox diam (mean±std)'}")
    print("-" * 80)

    for class_id in sorted(stats.keys()):
        values = stats[class_id]

        count = values["count"]
        refl = np.array(values["reflections"])
        bbox = np.array(values["bbox_diam"])

        name = class_names[class_id] if class_names else str(class_id)

        print(f"{name:<10} "
              f"{count:<10} "
              f"{refl.mean():.2f} ± {refl.std():.2f}{'':<5} "
              f"{bbox.mean():.2f} ± {bbox.std():.2f}")


def plot_bbox_boxplot(stats, class_names):
    # Sort by class id for consistent ordering
    sorted_ids = sorted(stats.keys())

    data = []
    labels = []

    for cid in sorted_ids:
        bbox = np.array(stats[cid]["bbox_diam"])
        data.append(bbox)
        labels.append(class_names[cid])

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.xlabel("Type")
    plt.ylabel("Bounding Box Diameter [m]")
    plt.title("Diameter Distribution by Type")
    plt.tight_layout()
    plt.show()

train_dataset = SingleFrameDataset(
    data_root_dir="/home/appuser/chrono_points_cls_benchmark",
    split="train",
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding"
)

val_dataset = SingleFrameDataset(
    data_root_dir="/home/appuser/chrono_points_cls_benchmark",
    split="val",
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding"
)

test_dataset = SingleFrameDataset(
    data_root_dir="/home/appuser/chrono_points_cls_benchmark",
    split="test",
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding"
)


all_datasets = [train_dataset, val_dataset, test_dataset]

combined_stats = compute_combined_stats(all_datasets)

class_names = {
    0: "P4",
    1: "M600",
    2: "Pigeon",
    3: "Hooded Crow",
    4: "Duck"
    
}

print_stats_table(combined_stats, class_names)

plot_bbox_boxplot(combined_stats, class_names)