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
import models.sequence_classification as Models
from data.dataset import PointSeriesDataset

from train_PSTNet import (
    FRAME_GAP_DICT,
    train_one_epoch,
    evaluate
)

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "/home/appuser/LIFT_benchmark"

NUM_FRAME = 12
SEED = 0 # TODO: SET TO 0

EPOCHS = 50

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
        num_workers=NUM_WORKERS
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

    Model = getattr(Models, 'MSRAction')
    model = Model(radius=0.5, nsamples=9, num_classes=5)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

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

    warmup_iters = WARMUP_EPOCHS * len(train_loader)
    lr_milestones = [len(train_loader) * m for m in LR_MILESTONES]
    lr_scheduler = utils.WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=LR_GAMMA, warmup_iters=warmup_iters, warmup_factor=1e-5)

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
            epoch=epoch,
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
        study_name="PSTNet_optimisation",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True
    )

    # study.set_metric_names(["val mAcc"])

    # --------------------------------------------------------
    # START OPTIMIZATION
    # --------------------------------------------------------

    study.optimize(
        objective,
        n_trials=N_TRIALS
    )

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