from __future__ import print_function
import datetime
import random
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import logging

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # Add the src directory to sys.path
from data.dataset import PointSeriesDataset

import utils

from scheduler import WarmupMultiStepLR

from datasets.msr import MSRAction3D
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

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq=None, has_pbar=True):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    metric_logger.add_meter('OA', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('mAcc', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))

    header = f'Epoch: [{epoch}]'

    num_classes = data_loader.dataset.num_types

    # epoch accumulators
    class_correct = torch.zeros(num_classes, device=device)
    class_count = torch.zeros(num_classes, device=device)
    total_correct = 0
    total_samples = 0
    
    # Only use MetricLogger when progress output is wanted.
    if has_pbar:
        data_iterator = metric_logger.log_every(
            data_loader, print_freq, header
        )
    else:
        data_iterator = data_loader

    for clip, target, _ in data_iterator:
        start_time = time.time()

        clip = clip.to(device)
        target = target.to(device)

        output = model(clip)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # predictions
        pred = output.argmax(dim=1)
        correct = (pred == target)

        total_correct += correct.sum().item()
        total_samples += target.size(0)

        # per-class stats
        for c in range(num_classes):
            mask = (target == c)
            class_count[c] += mask.sum()
            class_correct[c] += (correct & mask).sum()

        # compute running metrics (safe division)
        OA = total_correct / max(total_samples, 1)
        class_acc = class_correct / (class_count + 1e-6)
        mAcc = class_acc.mean().item()

        batch_size = clip.shape[0]

        if has_pbar:
            metric_logger.update(
                loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"]
            )
            metric_logger.meters['OA'].update(OA, n=batch_size)
            metric_logger.meters['mAcc'].update(mAcc, n=batch_size)
            metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))

        lr_scheduler.step()
        sys.stdout.flush()

    # final epoch stats
    final_OA = total_correct / max(total_samples, 1)
    final_class_acc = class_correct / (class_count + 1e-6)
    final_mAcc = final_class_acc.mean().item()

    if has_pbar:
        print(f' * Train OA {final_OA:.4f} mAcc {final_mAcc:.4f}')
        logging.getLogger().info(f' EPOCH [{epoch}] | Train OA {final_OA:.4f} mAcc {final_mAcc:.4f}')


def evaluate(model, criterion, data_loader, device, has_pbar=True):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('OA', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))
    metric_logger.add_meter('mAcc', utils.SmoothedValue(window_size=10, fmt='{value:.4f}'))

    header = 'Test:'

    num_classes = data_loader.dataset.num_types

    class_correct = torch.zeros(num_classes, device=device)
    class_count = torch.zeros(num_classes, device=device)
    total_correct = 0
    total_samples = 0
    
    if has_pbar:
        data_iterator = metric_logger.log_every(
            data_loader, 100, header
        )
    else:
        data_iterator = data_loader

    with torch.no_grad():
        for clip, target, _ in data_iterator:

            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(clip)
            loss = criterion(output, target)

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

            batch_size = clip.shape[0]

            if has_pbar:
                metric_logger.update(loss=loss.item())
                metric_logger.meters['OA'].update(OA, n=batch_size)
                metric_logger.meters['mAcc'].update(mAcc, n=batch_size)


    final_OA = total_correct / max(total_samples, 1)
    final_class_acc = class_correct / (class_count + 1e-6)
    final_mAcc = final_class_acc.mean().item()

    if has_pbar:
        metric_logger.synchronize_between_processes()
        print(f' * Eval OA {final_OA:.4f} mAcc {final_mAcc:.4f}')
        logging.getLogger().info(f' * Eval OA {final_OA:.4f} mAcc {final_mAcc:.4f}')

    return final_mAcc

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)
        logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), level=logging.INFO,\
                            format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger()

        #os.system('cp %s %s' % ('./modules/intra_mamba.py', args.output_dir))
        os.system('cp %s %s' % ('./models/msr.py', args.output_dir))
        os.system('cp %s %s' % ('./train-msr.py', args.output_dir))
        os.system('cp %s %s' % ('/usr/local/lib/python3.8/dist-packages/mamba_ssm/ops/selective_scan_interface.py', args.output_dir))

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    logging.getLogger().info("Arguments:")
    for key, value in vars(args).items():
        logging.getLogger().info("  %s: %s", key, value)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    # dataset = MSRAction3D(
            # root=args.data_path,
            # frames_per_clip=args.clip_len,
            # frame_interval=args.frame_interval,
            # num_points=args.num_points,
            # train=True
    # )
    # 
    dataset = PointSeriesDataset(
        data_root_dir='/home/appuser/LIFT_benchmark',
        split='train',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=args.config
    )

    # dataset_test = MSRAction3D(
    #         root=args.data_path,
    #         frames_per_clip=args.clip_len,
    #         frame_interval=args.frame_interval,
    #         num_points=args.num_points,
    #         train=False
    # )
    dataset_test = PointSeriesDataset(
        data_root_dir='/home/appuser/LIFT_benchmark',
        split='val',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=args.config
    )
    
    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim,mlp_dim=args.mlp_dim, num_classes=dataset.num_types,
                  depth_mamba_inter=args.depth_mamba_inter, rms_norm=args.rms_norm,
                  drop_out_in_block=args.drop_out_in_block, drop_path=args.drop_path,
                  depth_mamba_intra=args.depth_mamba_intra, intra=args.intra)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if torch.cuda.device_count() > 1:
        print(f'Number of GPUs: {torch.cuda.device_count()}') 
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1


    print(f"Start training for config {args.config}")

    if args.output_dir:
        logging.info('Device: %s', device)
        logging.info('Start training')

    start_time = time.time()
    acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)

        evaluate_acc = evaluate(model, criterion, data_loader_test, device=device)

        if evaluate_acc >= acc and args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            #utils.save_on_master(
                #checkpoint,
                #os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
            logging.info('Epoch %d new model saved, accuracy %f'%(epoch, evaluate_acc))
        acc = max(acc, evaluate_acc)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Accuracy {}'.format(acc))
    logger.info('Training time {}'.format(total_time_str))
    logger.info('Accuracy {}'.format(acc))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')
    
    parser.add_argument('--data-path', default='/dataset/MSR/MSR', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed', required=True)
    parser.add_argument('--model', default='MAMBA4D', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=12, type=int, metavar='N', help='number of frames per clip', required=True)
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
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')#0.01
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')#1e-4
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')
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

if __name__ == "__main__":
    args = parse_args()
    main(args)
