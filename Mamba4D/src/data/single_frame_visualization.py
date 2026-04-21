import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataset import SingleFrameDataset
import torch
import os



def plot_single_pointcloud(points,
                           title=None,
                           point_size=50,
                           elev=25,
                           azim=120,
                           save_path=None):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z,
               s=point_size,
               c='gray',
               alpha=0.8,)

    # Equal aspect ratio
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    
    x_min = mid_x-max_range
    y_min = mid_y-max_range
    z_min = mid_z-max_range
    
    x_max = mid_x+max_range
    y_max = mid_y+max_range
    z_max = mid_z+max_range
    

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.view_init(elev=elev, azim=azim)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel("X", fontsize=24)
    ax.set_ylabel("Y", fontsize=24)
    ax.set_zlabel("Z", fontsize=24)

    
    ax.grid(True)

    if title:
        ax.set_title(title, fontsize=28, pad=-5)


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ==========================
# Load dataset
# ==========================

test_dataset = SingleFrameDataset(
    data_root_dir="/home/appuser/chrono_points_cls_benchmark",
    split="train",
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    statistics=True,
    bbox_mode=0
)


# ==========================
# Plot samples
# ==========================

output_dir = "pcl_plots"
os.makedirs(output_dir, exist_ok=True)

for idx in range(len(test_dataset)):

    points, label, orig, diag_length = test_dataset[idx]
    

    
    # Remove zero padding
    if isinstance(orig, torch.Tensor):
        orig = orig.cpu()

    valid_mask = (orig != 0).any(axis=1)
    xyz = orig[valid_mask][:, :3]

    if xyz.shape[0] == 0:
        continue
    
    

    
    if label != 7 or xyz.shape[0] < 50:
        continue

    num_points = xyz.shape[0]

    save_path = None

    plot_single_pointcloud(
        xyz,
        title=f"{num_points} points",
        save_path=save_path
    )

    print(f"Saved {save_path}")