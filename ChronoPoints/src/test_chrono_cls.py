import os
import sys
from pathlib import Path
import numpy as np
import random
import torch
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import PointSeriesDataset
from trajectory_cls.chronopoints_cls import ChronoPointsClassifier
from logger import setup_logger
from tqdm import tqdm


FRAME_GAP_DICT = {
    2: 0, 4: 0, 8: 0,
    10: 1, 12: 1, 14: 1,
    16: 2, 18: 2, 20: 2,
}


# =========================
# Evaluation
# =========================
def evaluate(classifier, dataloader, device, logger, label_decoder=None):
    classifier.eval()

    total_correct = 0
    total_samples = 0

    num_classes = classifier.classifier[-1].out_features

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    confusion_counts = torch.zeros((num_classes, num_classes), dtype=torch.long)

    softmax = torch.nn.Softmax(dim=1)
    conf_correct_sum = 0.0
    conf_incorrect_sum = 0.0
    correct_count = 0
    incorrect_count = 0

    with torch.no_grad():
        for pcl_seq, masks, velocities, stamps, class_labels, type_labels in tqdm(dataloader):

            pcl_sequence = pcl_seq.to(device, non_blocking=True)
            mask = masks.to(device, non_blocking=True)
            velocities = velocities.to(device, non_blocking=True)
            labels = type_labels.to(device, non_blocking=True)

            logits, _ = classifier(pcl_sequence, mask, velocities)
            preds = logits.argmax(dim=1)

            probs = softmax(logits)
            confs = probs.max(dim=1).values

            is_correct = (preds == labels)

            conf_correct_sum += confs[is_correct].sum().item()
            conf_incorrect_sum += confs[~is_correct].sum().item()

            correct_count += is_correct.sum().item()
            incorrect_count += (~is_correct).sum().item()

            total_correct += is_correct.sum().item()
            total_samples += labels.size(0)

            for c_true in range(num_classes):
                mask_c = (labels == c_true)
                if mask_c.any():
                    class_correct[c_true] += (preds[mask_c] == c_true).sum().item()
                    class_total[c_true] += mask_c.sum().item()
                    for c_pred in range(num_classes):
                        confusion_counts[c_true, c_pred] += (preds[mask_c] == c_pred).sum().item()

    overall_acc = total_correct / total_samples
    class_acc = class_correct.float() / class_total.clamp(min=1)
    mean_acc = class_acc.mean().item()

    avg_conf_correct = conf_correct_sum / max(correct_count, 1)
    avg_conf_incorrect = conf_incorrect_sum / max(incorrect_count, 1)

    # logger.info("\n")
    logger.info("Model Confidence Summary")
    logger.info("----------------------------------------------------")
    logger.info(f"Correct predictions:   {correct_count:6d} | Avg confidence: {avg_conf_correct:.4f}")
    logger.info(f"Incorrect predictions: {incorrect_count:6d} | Avg confidence: {avg_conf_incorrect:.4f}")
    logger.info("----------------------------------------------------\n")

    logger.info("Per-Class Results:")
    logger.info("----------------------------------------------------")
    for c in range(num_classes):
        total_c = class_total[c].item()
        correct_c = class_correct[c].item()
        incorrect_c = total_c - correct_c

        class_name = label_decoder[c] if label_decoder else f"Class {c}"
        acc_c = correct_c / max(total_c, 1)

        logger.info(f"{class_name:15s} | Total: {total_c:5d} | Correct: {correct_c:5d} | Incorrect: {incorrect_c:5d} | Acc: {acc_c:.3f}")
    logger.info("----------------------------------------------------\n")

    logger.info("Prediction Distribution per True Class:")
    logger.info("----------------------------------------------------")
    for c_true in range(num_classes):

        class_name = label_decoder[c_true] if label_decoder else f"Class {c_true}"
        logger.info(f"{class_name}:")

        for c_pred in range(num_classes):
            count = confusion_counts[c_true, c_pred].item()
            if count > 0:
                pred_name = label_decoder[c_pred] if label_decoder else f"Class {c_pred}"
                logger.info(f"   {pred_name:15s} : {count}")

    logger.info("----------------------------------------------------\n")

    return overall_acc, mean_acc


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="ChronoPoints Testing")

    parser.add_argument('--run_dir', type=str, required=True, help='Path to run_dir to load best model from')

    return parser.parse_args()


def log_empty_line(logger):
    for handler in logger.handlers:
        if hasattr(handler, "stream"):
            handler.stream.write("\n")
            handler.flush()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    run_dir = Path("/home/appuser/src/output/" + args.run_dir)

    # -------------------------
    # Setup logger
    # -------------------------
    log_file = run_dir / "test.log"
    logger = setup_logger(log_file)

    logger.info(f"Testing run: {run_dir}")

    # -------------------------
    # Load checkpoint
    # -------------------------
    checkpoint_path = run_dir / "best_model.pth"
    assert checkpoint_path.exists(), "No best model checkpoint found!"
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    log_empty_line(logger)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.load(checkpoint_path, map_location=device)

    train_args = state['args']
    if not isinstance(train_args, dict):
        train_args = vars(train_args)

    logger.info("Training args:")
    for k, v in train_args.items():
        logger.info(f"{k}: {v}")
        
    log_empty_line(logger)

    # -------------------------
    # Reconstruct config
    # -------------------------
    num_frame = train_args['num_frame']
    num_gap = FRAME_GAP_DICT[num_frame]
    config = f"f{num_frame}g{num_gap}"

    # -------------------------
    # Fix randomness
    # -------------------------
    seed = train_args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -------------------------
    # Dataset
    # -------------------------
    test_dataset = PointSeriesDataset(
        data_root_dir=train_args['data_root'],
        split='test',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=config
    )

    label_encoder = test_dataset.type_encoder
    label_decoder = {v: k for k, v in label_encoder.items()}

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_args['batch_size'],
        shuffle=False,
        num_workers=train_args['num_workers']
    )

    # -------------------------
    # Model
    # -------------------------
    num_classes = test_dataset.num_types

    classifier = ChronoPointsClassifier(
        num_classes=num_classes,
        emb_dim=128,
        traj_T=num_frame,
        fusion_hidden_factor=4,
        fusion_dropout=0.4,
        centroid_mode="bbox",
        poly_order=3
    )

    classifier.load_state_dict(state['model_state_dict'])
    classifier.to(device)

    logger.info(f"Loaded model with val OA: {state['val_OA']}, val mAcc: {state['val_mAcc']}\n")

    # -------------------------
    # Evaluate
    # -------------------------
    test_OA, test_mAcc = evaluate(
        classifier, test_dataloader, device, logger, label_decoder
    )

    logger.info(f"Test Overall Accuracy: {test_OA*100:.2f}%")
    logger.info(f"Test Mean Accuracy: {test_mAcc*100:.2f}%")


if __name__ == '__main__':
    main()