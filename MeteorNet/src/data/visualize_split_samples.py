from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

class Arrow3D(FancyArrowPatch):
    def __init__(self, start, end, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._start = np.array(start)
        self._end   = np.array(end)

    def draw(self, renderer):
        # Project 3D coordinates to 2D
        x0, y0, z0 = self._start
        x1, y1, z1 = self._end
        x0_proj, y0_proj, _ = proj3d.proj_transform(x0, y0, z0, self.axes.get_proj())
        x1_proj, y1_proj, _ = proj3d.proj_transform(x1, y1, z1, self.axes.get_proj())
        self.set_positions((x0_proj, y0_proj), (x1_proj, y1_proj))
        super().draw(renderer)

    def do_3d_projection(self):
        """Required for 3D axes compatibility."""
        x0, y0, z0 = self._start
        x1, y1, z1 = self._end
        _, _, zs = proj3d.proj_transform([x0, x1], [y0, y1], [z0, z1], self.axes.get_proj())
        return np.min(zs)  # depth for z-sorting



# Path to your JSON file
json_path = Path("/home/appuser/chrono_points_cls_benchmark/train_test_split_v2/single_return/shuffled_train_file_list.json")

display_arrow = False

# Base directory for point cloud files (adjust if needed)
base_dir = Path("/home/appuser/chrono_points_cls_benchmark")

# Load JSON
with open(json_path, "r") as f:
    groups = json.load(f)

cmap = plt.get_cmap("tab20")

groups = groups[23321:]

for group_idx, group in tqdm(enumerate(groups), total=len(groups)):
    # Skip group if it has less than 15 non-PAD samples
    non_pad_count = sum(1 for file_rel in group if file_rel != "<PAD/>")
    if non_pad_count < 15:
        continue

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    first_stamp = None
    _break = False
    first_points, last_points = None, None


    for pcl_idx, file_rel in enumerate(group):
        if file_rel == "<PAD/>":
            continue

        file_path = base_dir / file_rel
        data = np.load(file_path)

        if "pcl" in data:
            points = data["pcl"]
            stamp = data["timestamp"]
            label = data["type"]
        else:
            continue

        if label != "M600":# or label == "P4" or label == "Pigeon":
            plt.close(fig)
            _break = True
            break

        if first_points is None:
            first_points = points
            first_stamp = stamp
        last_points = points  # will end up as the last valid frame

        color = cmap(pcl_idx % 20)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, color=color, alpha=1.0,
                label=f"frame {pcl_idx} ({int((stamp - first_stamp)/1000000)} ms)")
        
    if _break:
        continue

    ax.set_title(f"{label}", fontsize=20)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if display_arrow:
        # --- Draw arrow after loop ---
        if first_points is not None and last_points is not None:
            start = first_points.mean(axis=0)[:3]
            end   = last_points.mean(axis=0)[:3]
            arrow = Arrow3D(start, end, mutation_scale=30, lw=2, arrowstyle="-|>", color="k")
            ax.add_artist(arrow)

    # -75 or -165
    ax.view_init(elev=20, azim=-255)
    plt.show()

    print("\n")