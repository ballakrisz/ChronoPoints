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
from calflops import calculate_flops

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
from data.dataset import PointSeriesDataset
import models.msr as Models

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

def compute_model_complexity(model, dataloader, device):

    model.eval()

    # ====================================================
    # Parameters
    # ====================================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ====================================================
    # Get sample input
    # ====================================================
    clip, _, _ = next(iter(dataloader))

    # use a SINGLE sample for latency benchmarking
    single_clip = clip[:1].to(device)

    # ====================================================
    # FLOPs / MACs / Params
    # ====================================================
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=tuple(single_clip.shape),
        output_as_string=False,
        output_precision=4
    )

    print("-------------------- SANITY CHECK --------------------")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("------------------------------------------------------")

    print("\n---------------- MODEL COMPLEXITY ----------------")
    print(f"FLOPs:  {flops / 1e9:.4f} GFLOPs")
    print(f"MACs:   {macs / 1e9:.4f} GMACs")
    print(f"Params: {params / 1e6:.4f} M")
    print("--------------------------------------------------")

    # ====================================================
    # Robust inference benchmarking
    # ====================================================

    warmup_iters = 20
    benchmark_iters = 100

    print("\nWarming up GPU...")

    with torch.no_grad():

        # --------------------------------
        # Warmup
        # --------------------------------
        for _ in range(warmup_iters):
            _ = model(single_clip)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # --------------------------------
        # Benchmark
        # --------------------------------
        timings = []

        for _ in range(benchmark_iters):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(single_clip)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            timings.append(end - start)

    timings = np.array(timings)

    # ====================================================
    # Statistics
    # ====================================================
    mean_ms = timings.mean() * 1000
    std_ms = timings.std() * 1000
    median_ms = np.median(timings) * 1000
    min_ms = timings.min() * 1000
    max_ms = timings.max() * 1000

    fps = 1000.0 / mean_ms

    print("\n---------------- INFERENCE SPEED ----------------")
    print(f"Input shape:      {tuple(single_clip.shape)}")
    print(f"Warmup iterations:{warmup_iters}")
    print(f"Benchmark runs:   {benchmark_iters}")
    print("")
    print(f"Mean latency:     {mean_ms:.3f} ms")
    print(f"Median latency:   {median_ms:.3f} ms")
    print(f"Std latency:      {std_ms:.3f} ms")
    print(f"Min latency:      {min_ms:.3f} ms")
    print(f"Max latency:      {max_ms:.3f} ms")
    print(f"Throughput:       {fps:.2f} FPS")
    print("--------------------------------------------------\n")

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
    print(f"class acc: {class_acc}")
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


def main(opt=None):
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--data-path', default='/scratch/HeheFan-data/MSR-Action3D', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--frame-interval', default=1, type=int, metavar='N', help='interval of sampled frames')
    parser.add_argument('--num-points', default=512, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.7, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=14, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='output', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--save_root_dir', type=str, default='/home/appuser/src/output', help='output folder where model is saved')
    parser.add_argument('--data_root', type=str, default='/home/appuser/LIFT_benchmark', help='data root directory')
    
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
    model = Model(radius=opt.radius, nsamples=opt.nsamples, spatial_stride=opt.spatial_stride,
                  temporal_kernel_size=opt.temporal_kernel_size, temporal_stride=opt.temporal_stride,
                  emb_relu=opt.emb_relu,
                  dim=opt.dim, depth=opt.depth, heads=opt.heads, dim_head=opt.dim_head,
                  mlp_dim=opt.mlp_dim, num_classes=dataset_test.num_types)
    
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
    
    compute_model_complexity(model, data_loader_test, device)
    
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