from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from tqdm import tqdm
import logging
import argparse
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
from data.dataset import PointSeriesDataset
import models.sequence_classification as Models

DATA_ROOT = "/home/appuser/LIFT_benchmark"

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

def test(model, data_loader, device, opt, output_dir):
    """Test the model and compute all metrics (OA, mAcc, precision, recall, F1, inference time)"""
    
    model.eval()
    num_classes = data_loader.dataset.num_types
    conf_mat = np.zeros((num_classes, num_classes))
    
    total_inference_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            clip, target, _ = data
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Measure inference time
            torch.cuda.synchronize()
            forward_time_start = time.perf_counter()
            
            output = model(clip)
            
            torch.cuda.synchronize()
            forward_time_end = time.perf_counter()
            
            batch_inference_time = forward_time_end - forward_time_start
            total_inference_time += batch_inference_time
            total_samples += target.size(0)
            
            _, predicted = torch.max(output.data, 1)
            
            for t, p in zip(target.view(-1), predicted.view(-1)):
                conf_mat[t.item(), p.item()] += 1.0
    
    # Convert to tensor and ensure float dtype
    confusion_counts = torch.tensor(conf_mat, dtype=torch.float32)
    tp = confusion_counts.diag()
    fp = confusion_counts.sum(dim=0) - tp
    fn = confusion_counts.sum(dim=1) - tp
    
    # ========================
    # Inference time
    # ========================
    avg_inference_time_per_sample = total_inference_time / max(total_samples, 1)
    avg_inference_time_ms = avg_inference_time_per_sample * 1000.0
    
    # ========================
    # Per-class metrics
    # ========================
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    
    # ========================
    # Macro metrics
    # ========================
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    
    # ========================
    # Micro metrics
    # ========================
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()
    
    micro_precision = tp_sum / (tp_sum + fp_sum).clamp(min=1)
    micro_recall = tp_sum / (tp_sum + fn_sum).clamp(min=1)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp(min=1e-8)
    
    # ========================
    # Accuracy metrics
    # ========================
    overall_acc = tp_sum / confusion_counts.sum()
    class_acc = tp / confusion_counts.sum(dim=1).clamp(min=1)
    mean_acc = class_acc.mean().item()
    
    # ========================
    # Print results
    # ========================
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    print("\n[Inference Performance]")
    print("-"*60)
    print(f"Total inference time:         {total_inference_time:.4f} s")
    print(f"Average inference per sample: {avg_inference_time_ms:.4f} ms")
    print(f"Throughput:                   {1000.0/avg_inference_time_ms:.2f} samples/s")
    
    print("\n[Per-Class Precision / Recall / F1]")
    print("-"*60)
    for c in range(num_classes):
        print(f"Class {c:3d} | P: {precision[c]:.3f} | R: {recall[c]:.3f} | F1: {f1[c]:.3f}")
    
    print("\n[Averaged Metrics]")
    print("-"*60)
    print(f"Overall Accuracy (OA):  {overall_acc:.4f}")
    print(f"Mean Accuracy (mAcc):   {mean_acc:.4f}")
    print("")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print("")
    print(f"Micro Precision:  {micro_precision:.4f}")
    print(f"Micro Recall:     {micro_recall:.4f}")
    print(f"Micro F1:         {micro_f1:.4f}")
    print("="*60 + "\n")
    
    # Log to file
    logging.info("="*60)
    logging.info("TEST RESULTS")
    logging.info("="*60)
    
    logging.info("\n[Inference Performance]")
    logging.info("-"*60)
    logging.info(f"Total inference time:         {total_inference_time:.4f} s")
    logging.info(f"Average inference per sample: {avg_inference_time_ms:.4f} ms")
    logging.info(f"Throughput:                   {1000.0/avg_inference_time_ms:.2f} samples/s")
    
    logging.info("\n[Per-Class Precision / Recall / F1]")
    logging.info("-"*60)
    for c in range(num_classes):
        logging.info(f"Class {c:3d} | P: {precision[c]:.3f} | R: {recall[c]:.3f} | F1: {f1[c]:.3f}")
    
    logging.info("\n[Averaged Metrics]")
    logging.info("-"*60)
    logging.info(f"Overall Accuracy (OA):  {overall_acc:.4f}")
    logging.info(f"Mean Accuracy (mAcc):   {mean_acc:.4f}")
    logging.info("")
    logging.info(f"Macro Precision:  {macro_precision:.4f}")
    logging.info(f"Macro Recall:     {macro_recall:.4f}")
    logging.info(f"Macro F1:         {macro_f1:.4f}")
    logging.info("")
    logging.info(f"Micro Precision:  {micro_precision:.4f}")
    logging.info(f"Micro Recall:     {micro_recall:.4f}")
    logging.info(f"Micro F1:         {micro_f1:.4f}")
    logging.info("="*60 + "\n")
    
    return {
        'overall_acc': overall_acc.item(),
        'mean_acc': mean_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision.item(),
        'micro_recall': micro_recall.item(),
        'micro_f1': micro_f1.item(),
        'avg_inference_ms': avg_inference_time_ms,
        'throughput': 1000.0/avg_inference_time_ms,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'confusion_matrix': conf_mat
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="Testing PSTNet")
    
    # Model args
    parser.add_argument('--model', default='MSRAction', type=str, help='model')
    parser.add_argument('--radius', default=0.5, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=9, type=int, help='number of neighbors for the ball query')
    
    # Data args
    parser.add_argument('--clip-len', default=16, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--num-points', default=512, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    
    # Path args
    parser.add_argument('--save_root_dir', type=str, default='/home/appuser/src/output', help='output folder where model is saved')
    parser.add_argument('--data_root', type=str, default='/home/appuser/LIFT_benchmark', help='data root directory')
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint path (overrides best_mAcc.pth)')
    
    opt = parser.parse_args()
    
    # Setup config
    num_gap = FRAME_GAP_DICT[opt.clip_len]
    config = f"f{opt.clip_len}g{num_gap}"
    opt.config = config
    
    # Create run directory like second file
    run_dir = os.path.join(opt.save_root_dir, f"{config}_seed_{opt.seed}")
    os.makedirs(run_dir, exist_ok=True)
    opt.save_root_dir = run_dir
    
    # Setup logging inside this folder
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filename=os.path.join(opt.save_root_dir, 'test.log'),
        level=logging.INFO
    )
    
    print(opt)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)
    
    # Set seeds
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.empty_cache()
    
    # ========================
    # Load test data
    # ========================
    print("\nLoading test data...")
    
    dataset_test = PointSeriesDataset(
        data_root_dir=opt.data_root,
        split='test',
        max_points_per_frame=opt.num_points,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=opt.config
    )
    
    print(f"Test dataset size: {len(dataset_test)}")
    print(f"Number of classes: {dataset_test.num_types}")
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.workers, 
        pin_memory=True
    )
    
    # ========================
    # Load model
    # ========================
    print("\nLoading model...")
    Model = getattr(Models, opt.model)
    model = Model(radius=opt.radius, nsamples=opt.nsamples, num_classes=dataset_test.num_types)
    
    if opt.resume:
        checkpoint = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {opt.resume}")
    else:
        # Load best model from run directory (like second file)
        best_model_path = os.path.join(opt.save_root_dir, 'best_mAcc.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded best model from {best_model_path}")
        else:
            print("ERROR: No model checkpoint found!")
            print(f"Expected: {best_model_path}")
            return
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # ========================
    # Run test
    # ========================
    print("\n" + "="*60)
    print("Starting evaluation on test set...")
    print("="*60)
    
    results = test(
        model=model,
        data_loader=data_loader_test,
        device=device,
        opt=opt,
        output_dir=opt.save_root_dir
    )
    
    print("\nTest complete!")
    print(f"Overall Accuracy (OA): {results['overall_acc']:.4f}")
    print(f"Mean Accuracy (mAcc):  {results['mean_acc']:.4f}")
    print(f"Macro F1:              {results['macro_f1']:.4f}")
    print(f"Micro F1:              {results['micro_f1']:.4f}")
    print(f"Average inference:     {results['avg_inference_ms']:.4f} ms")
    print(f"Throughput:            {results['throughput']:.2f} samples/s")
    
    return results


if __name__ == "__main__":
    main()