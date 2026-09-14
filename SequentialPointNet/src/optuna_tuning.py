# optuna_tuning.py

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import optuna
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import random

from optuna.visualization import (
    plot_parallel_coordinate,
    plot_param_importances,
    plot_optimization_history,
    plot_slice,
    plot_contour
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import PointNet_Plus 
from utils import group_points_4DV_T_S
from data.dataset import PointSeriesDataset

from train import (
    FRAME_GAP_DICT
)

def train_one_epoch(netR, criterion, optimizer, scheduler, train_loader, device, epoch):
    # switch to train mode
    torch.cuda.synchronize()
    netR.train()
    loss_sigma = 0.0
    conf_mat = np.zeros((opt.Num_Class, opt.Num_Class))
    
    for i, data in enumerate(train_loader, 0):
        if len(data[0])==1:
            continue
        torch.cuda.synchronize()
        # 1 load imputs and target
        ## 3DV points and 3 temporal segment appearance points
        ## points_xyzc: B*2048*8;points_1xyz:B*2048*3  target: B*1
        points4DV_T,label,v_name = data
        points4DV_T,label = points4DV_T.cuda(),label.cuda()
        # print('points4DV_T:',points4DV_T.shape)
        xt, yt = group_points_4DV_T_S(points4DV_T, opt)#B*F*4*Cen*K  B*F*4*Cen*1
        # print('xt:',xt.shape)
        xt = xt.type(torch.FloatTensor)
        yt = yt.type(torch.FloatTensor)

        prediction = netR(xt, yt)
        
        loss = criterion(prediction,label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # update training error
        loss_sigma += loss.item()
        #_, predicted60 = torch.max(prediction.data[:,0:60], 1)
        _, predicted = torch.max(prediction.data, 1)
        # print(predicted.data)
        for t, p in zip(label.view(-1), predicted.view(-1)):
            conf_mat[t.item(), p.item()] += 1
            
    scheduler.step(epoch)
            
            
def evaluate(netR, criterion, val_loader, device, epoch):
    torch.cuda.synchronize()
    netR.eval()

    conf_mat = np.zeros((opt.Num_Class, opt.Num_Class))
    loss_sigma = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            points4DV_T, label, v_name = data
            points4DV_T, label = points4DV_T.cuda(), label.cuda()

            xt, yt = group_points_4DV_T_S(points4DV_T, opt)
            xt = xt.type(torch.FloatTensor)
            yt = yt.type(torch.FloatTensor)

            prediction = netR(xt, yt)
            loss = criterion(prediction, label)

            _, predicted = torch.max(prediction.data, 1)

            loss_sigma += loss.item()

            for t, p in zip(label.view(-1), predicted.view(-1)):
                conf_mat[t.item(), p.item()] += 1

    OA = conf_mat.trace() / conf_mat.sum()

    class_acc = []
    for i in range(opt.Num_Class):
        if conf_mat[i].sum() == 0:
            class_acc.append(0)
        else:
            class_acc.append(conf_mat[i, i] / conf_mat[i].sum())

    mAcc = np.mean(class_acc)
    
    return mAcc


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description = "Training")

    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')#￥￥￥￥
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--temperal_num', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--pooling', type=str, default='concatenation', help='how to aggregate temporal split features: vlad | concatenation | bilinear')
    parser.add_argument('--dataset', type=str, default='ntu60', help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')#￥￥￥￥
    parser.add_argument('--gamma', type=float, default=0.5, help='')#￥￥￥￥
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=0, required=False)

    parser.add_argument('--root_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\01_MSR3D',  help='preprocess folder')
    # parser.add_argument('--depth_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\paper\\dataset\\Prosessed_dataset\\01_MSR3D\\',  help='raw_depth_png')
    ################
    # parser.add_argument('--save_root_dir', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\code\\3DV-Action-master\\models\\ntu60\\xsub',  help='output folder')
    parser.add_argument('--save_root_dir', type=str, default='/home/appuser/src/output',  help='output folder')
    parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
    
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

    ########
    parser.add_argument('--Seg_size', type=int, default =1,  help='number of frame in seg')
    parser.add_argument('--stride', type=int, default = 1,  help='stride of seg')
    parser.add_argument('--all_framenum', type=int, default = 12,  help='number of action frame')
    parser.add_argument('--framenum', type=int, default = 12, required=False, help='number of action frame')
    parser.add_argument('--EACH_FRAME_SAMPLE_NUM', type=int, default = 512,  help='number of sample points in each frame')
    parser.add_argument('--T_knn_K', type=int, default = 48,  help='K for knn search of temperal stream')
    parser.add_argument('--T_knn_K2', type=int, default = 16,  help='K for knn search of temperal stream')
    parser.add_argument('--T_sample_num_level1', type=int, default = 128,  help='number of first layer groups')
    parser.add_argument('--T_sample_num_level2', type=int, default = 32,  help='number of first layer groups')
    parser.add_argument('--T_ball_radius', type=float, default=0.2, help='square of radius for ball query of temperal stream')
    
    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default = 2048,  help='number of sample points')

    parser.add_argument('--Num_Class', type=int, default = 10,  help='number of outputs')
    parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.1, help='square of radius for ball query in level 1')#0.025 -> 0.05 for detph
    parser.add_argument('--ball_radius2', type=float, default=0.2, help='square of radius for ball query in level 2')# 0.08 -> 0.01 for depth


    opt = parser.parse_args()
    opt.Num_Class = 5
    return opt

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "/home/appuser/LIFT_benchmark"

NUM_FRAME = 12
SEED = 0

EPOCHS = 30

BATCH_SIZE = 16
NUM_WORKERS = 4

N_TRIALS = 50


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gap = FRAME_GAP_DICT[NUM_FRAME]
config = f"f{NUM_FRAME}g{num_gap}"

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = Path(f"./optuna_results_{config}")
OUTPUT_DIR.mkdir(exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "best_models"
MODEL_DIR.mkdir(exist_ok=True)

DB_PATH = OUTPUT_DIR / "optuna.db"

# ============================================================
# FIX RANDOMNESS
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# NOTE:
# cuDNN BatchNorm is unexpectedly very slow for this model on our setup
# (PyTorch 2.3.1 / CUDA 11.8 / cuDNN 8.7), increasing training time from
# ~30 s/epoch to ~3 min/epoch. Profiling attributes the slowdown to
# cudnn_batch_norm. Disabling cuDNN forces PyTorch's native CUDA BatchNorm,
# restoring the expected performance without modifying the original model

# NOTE:
# If i change the first BatchNorm2d operation to GroupNorm and USE the cuDNN kernels, then a single epoch runs in 30s
# However, changing the model would weaken the comparioson with ChronoPoints, so its safer to just disable cuDNN altogether.
torch.backends.cudnn.enabled = False

# ============================================================
# DATASETS
# ============================================================

train_dataset = PointSeriesDataset(
    data_root_dir=DATA_ROOT,
    split='train',
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    sequence_format=config
)

val_dataset = PointSeriesDataset(
    data_root_dir=DATA_ROOT,
    split='val',
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    sequence_format=config
)

# ============================================================
# OBJECTIVE
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # HYPERPARAMETERS
    # --------------------------------------------------------

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-2,
        log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-1,
        log=True
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["SGD", "Adam", "AdamW"]
    )

    if optimizer_name == "SGD":
        momentum = trial.suggest_float(
            "momentum",
            0.5,
            0.99
        )
    else:
        momentum = None
    
    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = PointNet_Plus(opt)
    model = torch.nn.DataParallel(model).cuda()

    # --------------------------------------------------------
    # LOSSES
    # --------------------------------------------------------

    cls_loss = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )

    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_mAcc = 0.0

    epoch_pbar = tqdm(range(EPOCHS), desc=f"Trial {trial.number}")

    for epoch in epoch_pbar:

        train_dataset.current_epoch = epoch

        train_one_epoch(
            netR=model,
            criterion=cls_loss,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            train_loader=train_loader,
            device=DEVICE,
            epoch=epoch,
        )

        val_mAcc = evaluate(
            netR=model,
            criterion=cls_loss,
            val_loader=val_loader,
            device=DEVICE,
            epoch=epoch
        )

        # ----------------------------------------------------
        # REPORT TO OPTUNA
        # ----------------------------------------------------

        trial.report(val_mAcc, epoch)

        # ----------------------------------------------------
        # SAVE BEST FOR THIS TRIAL
        # ----------------------------------------------------

        if val_mAcc > best_val_mAcc:

            best_val_mAcc = val_mAcc

            save_path = MODEL_DIR / f"trial_{trial.number}_best.pth"

            torch.save(
                {
                    "trial": trial.number,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mAcc": val_mAcc,
                    "params": trial.params
                },
                save_path
            )

        # ----------------------------------------------------
        # PRUNING
        # ----------------------------------------------------

        if trial.should_prune():

            print(
                f"[TRIAL {trial.number}] "
                f"PRUNED at epoch {epoch}"
            )

            raise optuna.TrialPruned()

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        # print(
        #     f"[TRIAL {trial.number}] "
        #     f"Epoch={epoch:03d} | "
        #     f"Train OA={train_OA:.4f} mAcc={train_mAcc:.4f} | "
        #     f"Val OA={val_OA:.4f} mAcc={val_mAcc:.4f} | "
        #     f"Best={best_val_mAcc:.4f}"
        # )
        
        epoch_pbar.set_postfix_str(
            f"Val mAcc={val_mAcc:.4f}"
        )

    return best_val_mAcc

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    opt = parse_args()

    # --------------------------------------------------------
    # SAMPLER
    # --------------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=SEED
    )

    # --------------------------------------------------------
    # PRUNER
    # --------------------------------------------------------

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=30
    )

    # --------------------------------------------------------
    # STUDY
    # --------------------------------------------------------

    study = optuna.create_study(
        direction="maximize",
        study_name="SeqPointNet_optimisation",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True
    )

    # study.set_metric_names(["val mAcc"])

    # --------------------------------------------------------
    # START OPTIMIZATION
    # --------------------------------------------------------

    TARGET_TRIALS = 50

    remaining_trials = max(0, TARGET_TRIALS - len(study.trials))

    if remaining_trials > 0:
        print(
            f"Existing trials: {len(study.trials)} | "
            f"Running: {remaining_trials}"
        )
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print(f"Already have {len(study.trials)} trials. Nothing to run.")

    # ========================================================
    # BEST TRIAL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("BEST TRIAL")
    print("=" * 80)

    best_trial = study.best_trial

    print(f"Best validation mAcc: {best_trial.value:.4f}")

    print("\nBest hyperparameters:\n")

    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(OUTPUT_DIR / "best_results.txt", "w") as f:

        f.write(f"Best validation mAcc: {best_trial.value}\n\n")

        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")

    # ========================================================
    # SAVE ALL TRIALS
    # ========================================================

    df = study.trials_dataframe()

    df.to_csv(
        OUTPUT_DIR / "all_trials.csv",
        index=False
    )

    # ========================================================
    # GENERATE VISUALIZATIONS
    # ========================================================

    print("\nGenerating Optuna visualizations...\n")

    # --------------------------------------------------------
    # PARALLEL COORDINATE
    # --------------------------------------------------------

    fig = plot_parallel_coordinate(
        study,
        params=[
            "learning_rate",
            "weight_decay",
            "optimizer",
            "momentum"
        ]
    )

    fig.write_html(
        PLOTS_DIR / "parallel_coordinate.html"
        )

    # --------------------------------------------------------
    # PARAM IMPORTANCE
    # --------------------------------------------------------

    fig = plot_param_importances(study)

    fig.write_html(
        PLOTS_DIR / "param_importance.html"
    )

    # --------------------------------------------------------
    # OPTIMIZATION HISTORY
    # --------------------------------------------------------

    fig = plot_optimization_history(study)

    fig.write_html(
        PLOTS_DIR / "optimization_history.html"
    )

    # --------------------------------------------------------
    # SLICE PLOTS
    # --------------------------------------------------------

    fig = plot_slice(
        study,
        params=[
            "learning_rate",
            "weight_decay",
            "optimizer",
            "momentum"
        ]
    )

    fig.write_html(
        PLOTS_DIR / "slice_plot.html"
    )

    # --------------------------------------------------------
    # CONTOUR PLOT
    # --------------------------------------------------------

    fig = plot_contour(
        study,
        params=[
            "learning_rate",
            "weight_decay"
        ]
    )

    fig.write_html(
        PLOTS_DIR / "lr_weight_decay_contour.html"
    )

    # ========================================================
    # DONE
    # ========================================================

    print("=" * 80)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 80)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Database saved to: {DB_PATH}")

    print("\nTo launch the Optuna dashboard:\n")

    print(
        f"optuna-dashboard sqlite:///{DB_PATH}"
    )