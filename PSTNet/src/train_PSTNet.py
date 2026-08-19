from __future__ import print_function
import datetime
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
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add the current file's directory to sys.path

import utils

from datasets.msr import MSRAction3D
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

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    loss_sigma = 0.0
    num_classes = data_loader.dataset.num_types
    conf_mat = np.zeros((num_classes, num_classes))
    
    for i, data in enumerate(tqdm(data_loader, 0)):
        clip, target, _ = data
        clip, target = clip.to(device), target.to(device)
        output = model(clip)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sigma += loss.item()
        _, predicted = torch.max(output.data, 1)
        
        for t, p in zip(target.view(-1), predicted.view(-1)):
            conf_mat[t.item(), p.item()] += 1
            
        lr_scheduler.step()
    
    OA = conf_mat.trace() / conf_mat.sum()
    class_acc = []
    for i in range(num_classes):
        if conf_mat[i].sum() == 0:
            class_acc.append(0)
        else:
            class_acc.append(conf_mat[i, i] / conf_mat[i].sum())
    mAcc = np.mean(class_acc)
    loss_avg = loss_sigma / len(data_loader)
    
    print(f'[TRAIN] Epoch {epoch} | OA: {OA:.4f} | mAcc: {mAcc:.4f} | Loss: {loss_avg:.4f}')
    

def evaluate(model, criterion, data_loader, device, epoch):
    model.eval()
    num_classes = data_loader.dataset.num_types
    conf_mat = np.zeros((num_classes, num_classes))
    loss_sigma = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            clip, target, _ = data
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)
            loss_sigma += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            
            for t, p in zip(target.view(-1), predicted.view(-1)):
                conf_mat[t.item(), p.item()] += 1
    
    OA = conf_mat.trace() / conf_mat.sum()
    class_acc = []
    for i in range(num_classes):
        if conf_mat[i].sum() == 0:
            class_acc.append(0)
        else:
            class_acc.append(conf_mat[i, i] / conf_mat[i].sum())
    mAcc = np.mean(class_acc)
    loss_avg = loss_sigma / len(data_loader)
    
    print(f'[VAL] Epoch {epoch} | OA: {OA:.4f} | mAcc: {mAcc:.4f} | Loss: {loss_avg:.4f}')
    
    return mAcc


def main(args):

    num_gap = FRAME_GAP_DICT[args.clip_len]
    config = f"f{args.clip_len}g{num_gap}"
    
    if args.output_dir:
        args.output_dir = f"{args.output_dir}/{config}_seed_{args.seed}"
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()


    dataset = PointSeriesDataset(
        data_root_dir=DATA_ROOT,
        split='train',
        max_points_per_frame=args.num_points,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=config
    )
    
    dataset_val = PointSeriesDataset(
        data_root_dir=DATA_ROOT,
        split='val',
        max_points_per_frame=args.num_points,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=config
    )

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, num_classes=dataset.num_types)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = utils.WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    best_mAcc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch)
        
        current_mAcc = evaluate(model, criterion, data_loader_val, device, epoch)
        
        if current_mAcc > best_mAcc:
            best_mAcc = current_mAcc
            if args.output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'best_mAcc.pth'))
                print(f"New best model saved with mAcc: {best_mAcc:.4f}")
        
        # if args.output_dir:
        #     checkpoint = {
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args}
        #     utils.save_on_master(
        #         checkpoint,
        #         os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        #     utils.save_on_master(
        #         checkpoint,
        #         os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Best mAcc: {:.4f}'.format(best_mAcc))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PSTNet Training')

    parser.add_argument('--data-path', default='data/MSR-Action3D', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='MSRAction', type=str, help='model')
    parser.add_argument('--radius', default=0.5, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=9, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--clip-len', default=16, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--frame-interval', default=1, type=int, metavar='N', help='interval between sampled frames')
    parser.add_argument('--num-points', default=512, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=35, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='output', type=str, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
