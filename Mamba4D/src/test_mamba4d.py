from __future__ import print_function
import sys
import numpy as np
import torch
import torch.utils.data
from torch import nn
import argparse

import logging

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # Add the src directory to sys.path
from data.dataset import PointSeriesDataset

import models.msr as Models

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


def evaluate(model, criterion, data_loader, device, label_decoder=None):
    model.eval()

    num_classes = data_loader.dataset.num_types

    total_correct = 0
    total_samples = 0

    class_correct = torch.zeros(num_classes, device=device)
    class_count = torch.zeros(num_classes, device=device)

    confusion_counts = torch.zeros((num_classes, num_classes), device=device)

    softmax = torch.nn.Softmax(dim=1)

    conf_correct_sum = 0.0
    conf_incorrect_sum = 0.0
    correct_count = 0
    incorrect_count = 0

    with torch.no_grad():
        for clip, target, _ in data_loader:

            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(clip)

            preds = logits.argmax(dim=1)
            probs = softmax(logits)
            confs = probs.max(dim=1).values

            correct = (preds == target)

            conf_correct_sum += confs[correct].sum().item()
            conf_incorrect_sum += confs[~correct].sum().item()

            correct_count += correct.sum().item()
            incorrect_count += (~correct).sum().item()

            total_correct += correct.sum().item()
            total_samples += target.size(0)

            for c_true in range(num_classes):
                mask = (target == c_true)
                if mask.any():
                    class_count[c_true] += mask.sum()
                    class_correct[c_true] += (preds[mask] == c_true).sum()

                    for c_pred in range(num_classes):
                        confusion_counts[c_true, c_pred] += (preds[mask] == c_pred).sum()

    # =========================
    # Accuracy
    # =========================
    overall_acc = total_correct / max(total_samples, 1)
    class_acc = class_correct / (class_count + 1e-6)
    mean_acc = class_acc.mean().item()

    avg_conf_correct = conf_correct_sum / max(correct_count, 1)
    avg_conf_incorrect = conf_incorrect_sum / max(incorrect_count, 1)

    # =========================
    # Precision / Recall / F1
    # =========================
    tp = confusion_counts.diag()
    fp = confusion_counts.sum(dim=0) - tp
    fn = confusion_counts.sum(dim=1) - tp

    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    tp_sum = tp.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()

    micro_precision = tp_sum / (tp_sum + fp_sum).clamp(min=1)
    micro_recall = tp_sum / (tp_sum + fn_sum).clamp(min=1)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp(min=1e-8)

    # =========================
    # PRINT EVERYTHING
    # =========================
    print("\n================ EVALUATION ================")
    print(f"Overall Accuracy (OA): {overall_acc:.4f}")
    print(f"Mean Accuracy (mAcc):  {mean_acc:.4f}")

    print("\n--- Confidence ---")
    print(f"Correct:   {correct_count} | Avg conf: {avg_conf_correct:.4f}")
    print(f"Incorrect: {incorrect_count} | Avg conf: {avg_conf_incorrect:.4f}")

    print("\n--- Per-Class Accuracy ---")
    for c in range(num_classes):
        total_c = int(class_count[c].item())
        correct_c = int(class_correct[c].item())
        acc_c = correct_c / max(total_c, 1)

        name = label_decoder[c] if label_decoder else f"Class {c}"
        print(f"{name:15s} | Total: {total_c:5d} | Correct: {correct_c:5d} | Acc: {acc_c:.3f}")

    print("\n--- Per-Class Precision / Recall / F1 ---")
    for c in range(num_classes):
        name = label_decoder[c] if label_decoder else f"Class {c}"
        print(f"{name:15s} | P: {precision[c]:.3f} | R: {recall[c]:.3f} | F1: {f1[c]:.3f}")

    print("\n--- Averages ---")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print("")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall:    {micro_recall:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")

    print("\n--- Confusion Matrix (counts) ---")
    print(confusion_counts)

    print("==========================================\n")

    return overall_acc, mean_acc


def main(ckpt):
    device = torch.device("cuda")
    
    print(f"Loading checkpoint: {ckpt}")
    checkpoint = torch.load(ckpt, map_location='cpu')
    
    args = checkpoint['args']

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Loading TEST dataset...")

    dataset_test = PointSeriesDataset(
        data_root_dir='/home/appuser/chrono_points_cls_benchmark',
        split='test',   
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=args.config
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True
    )

    print("Creating model...")
    Model = getattr(Models, args.model)

    model = Model(
        radius=args.radius,
        nsamples=args.nsamples,
        spatial_stride=args.spatial_stride,
        temporal_kernel_size=args.temporal_kernel_size,
        temporal_stride=args.temporal_stride,
        emb_relu=args.emb_relu,
        dim=args.dim,
        mlp_dim=args.mlp_dim,
        num_classes=dataset_test.num_types,
        depth_mamba_inter=args.depth_mamba_inter,
        rms_norm=args.rms_norm,
        drop_out_in_block=args.drop_out_in_block,
        drop_path=args.drop_path,
        depth_mamba_intra=args.depth_mamba_intra,
        intra=args.intra
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)


    model.load_state_dict(checkpoint['model'])

    criterion = nn.CrossEntropyLoss()

    print("Running TEST evaluation...")
    evaluate(model, criterion, data_loader_test, device)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    
    parser.add_argument('--seed', default=0, type=int, help='random seed', required=True)
    parser.add_argument('--clip-len', default=12, type=int, metavar='N', help='number of frames per clip', required=True)

    args = parser.parse_args()
    
    args.config = "f{}g{}".format(args.clip_len, FRAME_GAP_DICT[args.clip_len])
    
    ckpt = f"/home/appuser/src/output/{args.config}_seed_{args.seed}/checkpoint.pth"

    return ckpt


if __name__ == "__main__":
    ckpt = parse_args()
    main(ckpt)