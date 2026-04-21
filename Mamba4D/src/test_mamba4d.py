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


def evaluate(model, criterion, data_loader, device):
    model.eval()

    num_classes = data_loader.dataset.num_types

    class_correct = torch.zeros(num_classes, device=device)
    class_count = torch.zeros(num_classes, device=device)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for clip, target, _ in data_loader:

            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(clip)

            pred = output.argmax(dim=1)
            correct = (pred == target)

            total_correct += correct.sum().item()
            total_samples += target.size(0)

            # per-class stats
            for c in range(num_classes):
                mask = (target == c)
                class_count[c] += mask.sum()
                class_correct[c] += (correct & mask).sum()

            # running metrics
            OA = total_correct / max(total_samples, 1)
            class_acc = class_correct / (class_count + 1e-6)
            mAcc = class_acc.mean().item()


    final_OA = total_correct / max(total_samples, 1)
    final_class_acc = class_correct / (class_count + 1e-6)
    final_mAcc = final_class_acc.mean().item()

    print(f' * Eval OA {final_OA:.4f} mAcc {final_mAcc:.4f}')
    logging.getLogger().info(f' * Eval OA {final_OA:.4f} mAcc {final_mAcc:.4f}')

    return final_mAcc


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