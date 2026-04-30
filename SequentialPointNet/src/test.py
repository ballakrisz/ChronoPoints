# -*- coding: utf-8 -*-
import torch
import os
import tqdm
import shutil
import collections
import argparse
import random
import time
#import gpu_utils as g
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1])) # Add the src directory to sys.path
from data.dataset import PointSeriesDataset

from model import PointNet_Plus#,Attension_Point,TVLAD
from utils import group_points_4DV_T_S

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

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

def main(args=None):
    parser = argparse.ArgumentParser(description = "Evaluation")

    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')#￥￥￥￥
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--temperal_num', type=int, default = 3,  help='number of input point features')
    parser.add_argument('--pooling', type=str, default='concatenation', help='how to aggregate temporal split features: vlad | concatenation | bilinear')
    parser.add_argument('--dataset', type=str, default='ntu60', help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')#￥￥￥￥
    parser.add_argument('--gamma', type=float, default=0.5, help='')#￥￥￥￥
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=0, required=True)

    parser.add_argument('--root_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\dataset\\Prosessed_dataset\\01_MSR3D',  help='preprocess folder')
    # parser.add_argument('--depth_path', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\paper\\dataset\\Prosessed_dataset\\01_MSR3D\\',  help='raw_depth_png')
    ################
    # parser.add_argument('--save_root_dir', type=str, default='C:\\Users\\Administrator\\Desktop\\LX\\paper\\code\\3DV-Action-master\\models\\ntu60\\xsub',  help='output folder')
    parser.add_argument('--save_root_dir', type=str, default='/home/appuser/src/output',  help='output folder')
    parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
    
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

    ########
    parser.add_argument('--Seg_size', type=int, default =1,  help='number of frame in seg')
    parser.add_argument('--stride', type=int, default = 1,  help='stride of seg')
    parser.add_argument('--all_framenum', type=int, default = 10,  help='number of action frame')
    parser.add_argument('--framenum', type=int, default = 10, required=True, help='number of action frame')
    parser.add_argument('--EACH_FRAME_SAMPLE_NUM', type=int, default = 512,  help='number of sample points in each frame')
    parser.add_argument('--T_knn_K', type=int, default = 48,  help='K for knn search of temperal stream')
    parser.add_argument('--T_knn_K2', type=int, default = 16,  help='K for knn search of temperal stream')
    parser.add_argument('--T_sample_num_level1', type=int, default = 128,  help='number of first layer groups')
    parser.add_argument('--T_sample_num_level2', type=int, default = 32,  help='number of first layer groups')
    parser.add_argument('--T_ball_radius', type=float, default=0.2, help='square of radius for ball query of temperal stream')
    
    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default = 2048,  help='number of sample points')

    parser.add_argument('--Num_Class', type=int, default = 10,  help='number of outputs')
    parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.1, help='square of radius for ball query in level 1')#0.025 -> 0.05 for detph
    parser.add_argument('--ball_radius2', type=float, default=0.2, help='square of radius for ball query in level 2')# 0.08 -> 0.01 for depth


    opt = parser.parse_args()
    print (opt)
    # torch.cuda.set_device(opt.main_gpu)

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    opt.config = "f{}g{}".format(opt.framenum, FRAME_GAP_DICT[opt.framenum])
    opt.all_framenum = opt. framenum

    # ===== Create run directory based on config =====
    run_dir = os.path.join(opt.save_root_dir, f"{opt.config}_seed_{opt.seed}")

    os.makedirs(run_dir, exist_ok=True)

    # update save path
    opt.save_root_dir = run_dir

    # logging inside this folder
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filename=os.path.join(opt.save_root_dir, 'test.log'),
        level=logging.INFO
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    #################改2#############
    data_val = PointSeriesDataset(
        data_root_dir='/home/appuser/chrono_points_cls_benchmark',
        split='test',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=opt.config
    )
    opt.Num_Class = data_val.num_types
    val_loader = DataLoader(dataset = data_val, batch_size = 8,num_workers = 8)

    #net =

    netR = PointNet_Plus(opt)
    #################改3#############
    netR.load_state_dict(torch.load(os.path.join(opt.save_root_dir, 'best_mAcc.pth')))

    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()
    print(netR)
    
    total_params = sum(p.numel() for p in netR.parameters())
    trainable_params = sum(p.numel() for p in netR.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    
    # evaluate mode
    torch.cuda.synchronize()
    netR.eval()

    conf_mat = np.zeros((opt.Num_Class, opt.Num_Class))

    with torch.no_grad(): 
        for i, data in enumerate(tqdm(val_loader)):
            torch.cuda.synchronize()

            points4DV_T, label, vid_name = data
            points4DV_T, label = points4DV_T.cuda(), label.cuda()

            xt, yt = group_points_4DV_T_S(points4DV_T, opt)
            xt = xt.type(torch.FloatTensor)
            yt = yt.type(torch.FloatTensor)

            forward_time_start = time.time()
            prediction = netR(xt, yt)
            forward_time_end = time.time()

            # print('forward time:', forward_time_end - forward_time_start)

            _, predicted = torch.max(prediction.data, 1)

            for j in range(len(label)):
                gt = label[j].item()
                pred = predicted[j].item()

                if pred != gt:
                    logging.info(
                        'Video Name:{} -- correct label {} predicted to {}'.format(
                            vid_name[j], gt, pred
                        )
                    )

                conf_mat[gt, pred] += 1.0


        confusion_counts = torch.tensor(conf_mat)

        tp = confusion_counts.diag()
        fp = confusion_counts.sum(dim=0) - tp
        fn = confusion_counts.sum(dim=1) - tp

        # Per-class metrics
        precision = tp.float() / (tp + fp).clamp(min=1)
        recall = tp.float() / (tp + fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        # Macro
        macro_precision = precision.mean().item()
        macro_recall = recall.mean().item()
        macro_f1 = f1.mean().item()

        # Micro
        tp_sum = tp.sum().float()
        fp_sum = fp.sum().float()
        fn_sum = fn.sum().float()

        micro_precision = tp_sum / (tp_sum + fp_sum).clamp(min=1)
        micro_recall = tp_sum / (tp_sum + fn_sum).clamp(min=1)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp(min=1e-8)

        # Accuracy
        overall_acc = tp_sum / confusion_counts.sum()
        class_acc = tp.float() / confusion_counts.sum(dim=1).clamp(min=1)
        mean_acc = class_acc.mean().item()
        
        logging.info("Per-Class Precision / Recall / F1:")
        logging.info("----------------------------------------------------")
        for c in range(opt.Num_Class):
            logging.info(
                f"Class {c:3d} | "
                f"P: {precision[c]:.3f} | R: {recall[c]:.3f} | F1: {f1[c]:.3f}"
            )
        logging.info("----------------------------------------------------\n")

        logging.info("Averaged Metrics:")
        logging.info("----------------------------------------------------")
        logging.info(f"Overall Accuracy: {overall_acc:.4f}")
        logging.info(f"Mean Accuracy:    {mean_acc:.4f}")
        logging.info("")
        logging.info(f"Macro Precision:  {macro_precision:.4f}")
        logging.info(f"Macro Recall:     {macro_recall:.4f}")
        logging.info(f"Macro F1:         {macro_f1:.4f}")
        logging.info("")
        logging.info(f"Micro Precision:  {micro_precision:.4f}")
        logging.info(f"Micro Recall:     {micro_recall:.4f}")
        logging.info(f"Micro F1:         {micro_f1:.4f}")
        logging.info("----------------------------------------------------\n")

if __name__ == '__main__':
    main()

