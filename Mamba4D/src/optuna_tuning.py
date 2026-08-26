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

import utils
import models.msr as Models
from data.dataset import PointSeriesDataset

from train_mambda4d import (
    FRAME_GAP_DICT,
    train_one_epoch,
    evaluate
)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    
    parser.add_argument('--data-path', default='/dataset/MSR/MSR', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed', required=False)
    parser.add_argument('--model', default='MAMBA4D', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=12, type=int, metavar='N', help='number of frames per clip', required=False)
    parser.add_argument('--frame-interval', default=1, type=int, metavar='N', help='interval of sampled frames')
    parser.add_argument('--num-points', default=512, type=int, metavar='N', help='number of points per frame')
    # intra-mamba
    parser.add_argument('--radius', default=0.7, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # training
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')#0.01
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')#1e-4
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # mamba
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth-mamba-inter', default=4, type=int)
    parser.add_argument('--depth-mamba-intra', default=12, type=int)
    parser.add_argument('--rms-norm', default=False, action='store_true')
    parser.add_argument('--drop-out-in-block', default=0.1, type=float)
    parser.add_argument('--drop-path', default=0.1, type=float)
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    parser.add_argument('--intra', default=True, type=bool, help='use intra-mamba or not')
    # patch
    parser.add_argument('--group-size', default=64, type=int)
    parser.add_argument('--num-group', default=32, type=int)
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output_dir', default=f"/home/appuser/src/output", type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()
    
    args.config = "f{}g{}".format(args.clip_len, FRAME_GAP_DICT[args.clip_len])
    args.output_dir = f"/home/appuser/src/output/{args.config}_seed_{args.seed}"
    

    return args

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "/home/appuser/LIFT_benchmark"

NUM_FRAME = 12
SEED = 0 # TODO: SET TO 0

EPOCHS = 30

BATCH_SIZE = 16
NUM_WORKERS = 4

N_TRIALS = 50

WARMUP_EPOCHS = 5
LR_MILESTONES = [20, 30]
LR_GAMMA = 0.1

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
        num_workers=NUM_WORKERS,
        pin_memory=True,
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

    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim,mlp_dim=args.mlp_dim, num_classes=dataset.num_types,
                  depth_mamba_inter=args.depth_mamba_inter, rms_norm=args.rms_norm,
                  drop_out_in_block=args.drop_out_in_block, drop_path=args.drop_path,
                  depth_mamba_intra=args.depth_mamba_intra, intra=args.intra)

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

    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_mAcc = 0.0

    epoch_pbar = tqdm(range(EPOCHS), desc=f"Trial {trial.number}")

    for epoch in epoch_pbar:

        train_dataset.current_epoch = epoch

        train_one_epoch(
            model=model,
            criterion=cls_loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            data_loader=train_loader,
            device=DEVICE,
            epoch=epoch,
            has_pbar=False
        )

        val_mAcc = evaluate(
            model=model,
            criterion=cls_loss,
            data_loader=val_loader,
            device=DEVICE,
            has_pbar=False
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
    args = parse_args()
    
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
        study_name="Mambda4D_optimisation",
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