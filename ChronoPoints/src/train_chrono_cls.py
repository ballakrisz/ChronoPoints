import os
import sys
from pathlib import Path
import numpy as np
import random
import torch
from datetime import datetime
import shutil
import argparse
from torchinfo import summary


sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add the current file's directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1])) # Add the src directory to sys.path

from trajectory_cls.chronopoints_utils import CrossEncoderContrastiveLoss
from data.dataset import PointSeriesDataset
from trajectory_cls.chronopoints_cls import ChronoPointsClassifier
from logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FRAME_GAP_DICT = {
    2 : 0,
    4 : 0,
    8 : 0,
    10 : 1,
    12 : 1,
    14 : 1,
    16 : 2,
    18 : 2,
    20 : 2,
}


def save_run_code(run_dir: Path):
    """
    Save only the training script and essential model/util files for reproducibility.
    """

    code_dir = run_dir / "code"
    code_dir.mkdir(exist_ok=True)

    # 1. Copy the training script (this file)
    this_script = Path(__file__).resolve()
    shutil.copy2(this_script, code_dir / this_script.name)

    # Base ChronoPoints path (adjust if needed)
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    print(base_dir)

    # 2. Copy ChronoPoints/trajectory_cls/chronopoints_cls.py
    cls_file = base_dir / "trajectory_cls" / "chronopoints_cls.py"
    if cls_file.exists():
        shutil.copy2(cls_file, code_dir / "chronopoints_cls.py")

    # 3. Copy ChronoPoints/trajectory_cls/chronopoints_utils.py
    utils_file = base_dir / "trajectory_cls" / "chronopoints_utils.py"
    if utils_file.exists():
        shutil.copy2(utils_file, code_dir / "chronopoints_utils.py")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_one_epoch(classifier, dataloader, cls_loss, contrastive_loss,contrastive_lambda, optimizer, scheduler, device, epoch, has_pbar=True):
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Per-class accuracy accumulators
    num_classes = classifier.classifier[-1].out_features
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    
    iterator = enumerate(dataloader)
    
    if has_pbar:
        iterator = tqdm(
            iterator,
            total=len(dataloader),
            desc=f"Epoch {epoch}"
        )

    for batch_id, (pcl_seq, masks, velocities, stamps, class_labels, type_labels) in iterator:
        pcl_sequence = pcl_seq.to(device, non_blocking=True)
        mask = masks.to(device, non_blocking=True)
        velocities = velocities.to(device, non_blocking=True)
        labels = type_labels.to(device, non_blocking=True)
    
        optimizer.zero_grad()

        logits, (p_dist,p_traj, p_spatio) = classifier(pcl_sequence, mask, velocities)

        # CE loss
        loss_cls = cls_loss(logits, labels)
        loss_contrastive = contrastive_loss(p_dist=p_dist, p_traj=p_traj, p_spatio=p_spatio)
        
        # combined objective
        loss = loss_cls + contrastive_lambda * loss_contrastive


        # Backprop
        loss.backward()
        optimizer.step()

        batch_size = pcl_sequence.size(0)
        total_loss += loss.item() * batch_size

        # Predictions
        preds = logits.argmax(dim=1)

        # Overall accuracy
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        # Per-class accuracy
        for c in range(num_classes):
            mask_c = (labels == c)
            if mask_c.any():
                class_correct[c] += (preds[mask_c] == c).sum().item()
                class_total[c] += mask_c.sum().item()

    avg_loss = total_loss / total_samples
    overall_acc = total_correct / total_samples

    # Mean per-class accuracy
    class_acc = class_correct.float() / class_total.clamp(min=1)
    mean_acc = class_acc.mean().item()

    # update scheduler per-epoch
    scheduler.step()

    return avg_loss, overall_acc, mean_acc

def evaluate(classifier, dataloader, loss_fn, device):
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Per-class stats
    num_classes = classifier.classifier[-1].out_features
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for pcl_seq, masks, velocities, stamps, class_labels, type_labels in dataloader:
            pcl_sequence = pcl_seq.to(device, non_blocking=True)
            mask = masks.to(device, non_blocking=True)
            velocities = velocities.to(device, non_blocking=True)
            labels = type_labels.to(device, non_blocking=True)

            logits, _ = classifier(pcl_sequence, mask, velocities)
            loss = loss_fn(logits, labels)

            batch_size = pcl_sequence.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)

            # Overall accuracy
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            # Per-class accuracy
            for c in range(num_classes):
                mask_c = (labels == c)
                if mask_c.any():
                    class_correct[c] += (preds[mask_c] == c).sum().item()
                    class_total[c] += mask_c.sum().item()

    avg_loss = total_loss / total_samples
    overall_acc = total_correct / total_samples

    # mean accuracy across classes
    class_acc = class_correct.float() / class_total.clamp(min=1)
    mean_acc = class_acc.mean().item()

    

    return avg_loss, overall_acc, mean_acc, class_acc

def parse_args():
    parser = argparse.ArgumentParser(description="ChronoPoints Training")

    # Hyperparams
    parser.add_argument('--epochs', type=int, default=200, help='Total number of trainign epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size used during training. Larger values are beneficial when using contrastive loss.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='Number of epochs between learning rate decay steps.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.7, help='Multiplicative factor applied during learning rate decay.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum allowed learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes used by the dataloaders')
    parser.add_argument('--output_dir', type=str, default="/home/appuser/src/output", help='Directory where training outputs (logs, checkpoints) are saved.')
    parser.add_argument('--val_frequency', type=int, default=1, help='Perform validation every val_frequency epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to a run directory to resume training from.')

    # Dataset
    parser.add_argument('--data_root', type=str, default='/home/appuser/chronopoints_cls_benchmark', help="Root directory of the chrono_points_cls_benchmark dataset")

    # Point cloud sequence
    parser.add_argument('--num_frame', type=int, default=None, help='Number of frames in each point cloud sequence.')

    # Contrastive loss
    parser.add_argument('--contrastive_lambda', type=float, default=0.3, help="The weight of which the contrastive loss is added to the cls loss")
    parser.add_argument('--temperature', type=float, default=0.07, help="Temperature scaling parameter for the contrastive loss")


    args = parser.parse_args()

    # validate args: either (--num_frame AND --seed) have to be provided OR (--resume)
    if args.resume is None:
        if args.seed is None or args.num_frame is None:
            parser.error("You must provide --seed AND --num_frame when NOT using --resume.")
    else:
        if args.seed is not None or args.num_frame is not None:
            print(" Warning: --resume ignores --seed and --num_frame (loaded from checkpoint).")

    return args

def log_empty_line(logger):
    for handler in logger.handlers:
        if hasattr(handler, "stream"):
            handler.stream.write("\n")
            handler.flush()

def main():
    args = parse_args()
    
    # -------------------------
    # RESUME OR NEW RUN
    # -------------------------
    if args.resume is not None:
        run_dir = Path("/home/appuser/src/output/" + args.resume)
        assert run_dir.exists(), "Resume directory does not exist"

        log_file = run_dir / "train.log"
        logger = setup_logger(log_file)

        logger.info(f"Resuming training from: {run_dir}")

        checkpoint_path = run_dir / "last_model.pth"
        assert checkpoint_path.exists(), "No last checkpoint found!"

        state = torch.load(checkpoint_path, map_location='cpu')

        saved_args = state['args']
        if not isinstance(saved_args, dict):
            saved_args = vars(saved_args)

        # restore args (overwrite current args)
        for k, v in saved_args.items():
            if k != "resume":
                setattr(args, k, v)

        start_epoch = state['epoch'] + 1
        best_val_mAcc = state.get('val_mAcc', 0.0)

        logger.info(f"Resumed from epoch {state['epoch']}")

        run_name = run_dir.name

    else:
        
        num_frame = args.num_frame
        num_gap = FRAME_GAP_DICT[num_frame]
        config = f"f{num_frame}g{num_gap}"

        run_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        run_dir = Path(args.output_dir) / f"{config}_seed_{args.seed}_{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log_file = run_dir / "train.log"
        logger = setup_logger(log_file)

        train_args = vars(args)

        logger.info("Training args:")
        for k, v in train_args.items():
            logger.info(f"{k}: {v}")
            
        log_empty_line(logger)    

        logger.info(f"Checkpoint directory: {run_dir}")

        save_run_code(run_dir)

        start_epoch = 0
        best_val_mAcc = 0.0

    # -------------------------
    # CONFIG FOR DATALOADER
    # -------------------------
    num_frame = args.num_frame
    num_gap = FRAME_GAP_DICT[num_frame]
    config = f"f{num_frame}g{num_gap}"

    # -------------------------
    # FIX RANDOMNESS
    # -------------------------
    seed_everything(args.seed)

    # TensorBoard
    tb_log_dir = run_dir / "tensorboard"
    tb_log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    val_freq = args.val_frequency 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # -------------------------
    # DATASETS
    # -------------------------
    train_dataset = PointSeriesDataset(
        data_root_dir=args.data_root,
        split='train',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=config
    )

    val_dataset = PointSeriesDataset(
        data_root_dir=args.data_root,
        split='val',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=config
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
        persistent_workers=True
    )

    # -------------------------
    # MODEL
    # -------------------------
    num_classes = train_dataset.num_types
    logger.info(f"Starting training with {num_classes} types.\n")

    classifier = ChronoPointsClassifier(
        num_classes=num_classes,
        emb_dim=128,
        traj_T=num_frame,
        fusion_hidden_factor=4,
        fusion_dropout=0.4,
        centroid_mode="bbox",
        poly_order=3
    )
    classifier.to(device)
    
    # Example dimensions
    B = args.batch_size   # batch size
    T = num_frame   # temporal frames
    N = 512 # points per frame

    # Dummy inputs
    pts = torch.randn(B, T, N, 3).to(device)

    # Boolean mask (True = valid point)
    mask = torch.ones(B, T, N, dtype=torch.bool).to(device)

    # Example velocities
    velocities = torch.randn(B, T, 3).to(device)

    model_summary = summary(
        classifier,
        input_data=(pts, mask, velocities),
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "trainable"
        ),
        depth=5,
        verbose=1
    )
    logger.info(f"MODEL SUMMARY\n{str(model_summary)}\n")

    # Loss
    cls_loss = torch.nn.CrossEntropyLoss()
    contrastive_loss = CrossEncoderContrastiveLoss(temperature=args.temperature).to(device)
    contrastive_lambda = args.contrastive_lambda

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(
            args.lr_decay_rate ** (epoch // args.lr_decay_step),
            args.min_lr / args.lr
        )
    )

    # -------------------------
    # LOAD STATE IF RESUMING
    # -------------------------
    if args.resume is not None:
        classifier.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])

    # -------------------------
    # TRAIN LOOP (UNCHANGED LOGIC)
    # -------------------------
    for epoch in range(start_epoch, args.epochs + 1):

        train_dataset.current_epoch = epoch

        train_loss, train_OA, train_mAcc = train_one_epoch(
            classifier, train_dataloader, cls_loss,
            contrastive_loss, contrastive_lambda,
            optimizer, scheduler, device, epoch
        )

        logger.info(f"[Epoch {epoch:03d}] "
                    f"Train: Loss={train_loss:.4f}, OA={train_OA:.4f}, mAcc={train_mAcc:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}")

        # -------------------------
        # VALIDATION (UNCHANGED)
        # -------------------------
        if epoch % val_freq == 0:

            val_loss, val_OA, val_mAcc, val_class_ass = evaluate(
                classifier, val_dataloader, cls_loss, device
            )

            logger.info(f"            Val:   Loss={val_loss:.4f}, OA={val_OA:.4f}, mAcc={val_mAcc:.4f}")
            logger.info(f"                   Class-wise={val_class_ass.tolist()}")

            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Val_OA', val_OA, epoch)
            writer.add_scalar('Accuracy/Val_mAcc', val_mAcc, epoch)

            if val_mAcc > best_val_mAcc:
                best_val_mAcc = val_mAcc

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mAcc': val_mAcc,
                    'val_OA': val_OA,
                    'args': vars(args)
                }, run_dir / "best_model.pth")

                logger.info(f"🔥 New Best Model Saved! mAcc={val_mAcc:.4f}")

        # -------------------------
        # SAVE LAST MODEL (USED FOR RESUMING TRAINING)
        # -------------------------
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_mAcc': best_val_mAcc,
            'args': vars(args)
        }, run_dir / "last_model.pth")

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train_OA', train_OA, epoch)
        writer.add_scalar('Accuracy/Train_mAcc', train_mAcc, epoch)

        log_empty_line(logger)

    logger.info("Training complete.")
    logger.info(f"Best validation Mean Accuracy: {best_val_mAcc*100:.2f}%")



if __name__ == '__main__':
    main()