import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add the current file's directory to sys.path

from tqdm import tqdm
from time import time
from dataset import PointSeriesDataset, SingleFrameDataset
from kfold_wrapper import PointSeriesKfoldWrapper

DATA_ROOT = '/home/appuser/chrono_points_cls_benchmark'

num_frames = 12
num_gaps = 1
config = f"f{num_frames}g{num_gaps}"


train_ds = PointSeriesDataset(
    data_root_dir=DATA_ROOT,
    split='train',
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    sequence_format=config
)
val_ds = PointSeriesDataset(
    data_root_dir=DATA_ROOT,
    split='val',
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    sequence_format=config
)

kfold = PointSeriesKfoldWrapper(
    train_dataset=train_ds,
    val_dataset=val_ds,
    n_splits=5,
    batch_size=32,
    num_workers=4,
    persistent_workers=True
)


for fold, train_loader, val_loader in kfold.folds():
    # start = time()
    # for i in range(10):
    #     for batch_id, (pcl_seq, masks, velocities, stamps, class_labels, type_labels) in tqdm(
    #         enumerate(train_loader), total=len(train_loader), desc=f"Train fold {fold}"
    #     ):
    #         s = 0
    # elapsed = time() - start
    # print(f"Total processing time for 10 epochs: {elapsed:.2f} seconds")
    # break
    # for batch_id, (pcl_seq, masks, velocities, stamps, class_labels, type_labels) in tqdm(
    #     enumerate(val_loader), total=len(val_loader), desc=f"Validation fold {fold}"
    # ):
    s = 0
    
    