import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

base_dir = "/home/appuser/chrono_points_cls_benchmark"

ordered_types = ["P4", "M600", "Pigeon", "Hooded crow", "Duck"]

stats = defaultdict(lambda: {
    "count": 0,
    "reflections": [],
    "bbox_diam": []
})

def normalize_type(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return normalize_type(value.item())
        else:
            return [normalize_type(v) for v in value.flatten()]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8").strip()
    if isinstance(value, (str, np.str_)):
        return value.strip()
    return value


for root, _, files in os.walk(base_dir):

    # skip double return
    if "double_return" in root:
        continue

    # only birds + single_return drones
    if not ("01503061" in root or "single_return" in root):
        continue

    for file in files:
        if not file.endswith(".npz"):
            continue

        path = os.path.join(root, file)

        try:
            with np.load(path, allow_pickle=False) as data:

                if "type" not in data or "pcl" not in data:
                    continue

                obj_type = normalize_type(data["type"])
                
                if obj_type not in ordered_types:
                    continue
                
                pcl = data["pcl"]

                if obj_type in ("", None):
                    continue

                # Remove zero padding if present
                valid_mask = (pcl[:, :3] != 0).any(axis=1)
                valid_points = pcl[valid_mask]

                if len(valid_points) == 0:
                    continue

                # Reflections = number of valid points
                reflections = len(valid_points)

                # Bounding box diagonal
                min_coords = valid_points[:, :3].min(axis=0)
                max_coords = valid_points[:, :3].max(axis=0)
                diag = np.linalg.norm(max_coords - min_coords)
                
                if obj_type == "P4" and diag > 2: 
                    continue
                
                if obj_type == "Pigeon" and diag > 1.2:
                    continue
                
                if obj_type == "M600" and diag > 1.5:
                    continue
                
                if obj_type == "Duck" and diag > 1.5:
                    continue

                stats[obj_type]["count"] += 1
                stats[obj_type]["reflections"].append(reflections)
                stats[obj_type]["bbox_diam"].append(diag)

        except Exception as e:
            print(f"Error reading {path}: {e}")


# =========================
# PRINT TABLE
# =========================

print(f"{'Type':<15} {'Samples':<10} {'Reflections (mean±std)':<25} {'Avg bbox diam (mean±std)'}")
print("-" * 90)

for t in ordered_types:
    if t not in stats:
        continue

    count = stats[t]["count"]
    refl = np.array(stats[t]["reflections"])
    bbox = np.array(stats[t]["bbox_diam"])

    print(f"{t:<15} "
          f"{count:<10} "
          f"{refl.mean():.2f} ± {refl.std():.2f}{'':<5} "
          f"{bbox.mean():.2f} ± {bbox.std():.2f}")


# =========================
# BOXPLOT (Correct Order)
# =========================

data = []
labels = []

for t in ordered_types:
    if t in stats:
        data.append(stats[t]["bbox_diam"])
        labels.append(t)
        
        
labels = ["P4", "M600", "Pigeon", "Hooded Crow", "Duck"]

plt.figure(figsize=(12,6))

plt.boxplot(data)

plt.xticks(
    ticks=range(1, len(labels) + 1),
    labels=labels,
    fontsize=22
)

plt.yticks(fontsize=20)

plt.ylabel("Bounding Box Diameter [m]", fontsize=22, labelpad=15)
plt.title("Diameter Distribution by Type", fontsize=24)

plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()