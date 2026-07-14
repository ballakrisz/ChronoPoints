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
    parser = argparse.ArgumentParser(description = "Training")

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

    # update save path to run-specific directory
    opt.save_root_dir = run_dir

    # setup logging inside this folder
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filename=os.path.join(opt.save_root_dir, 'train.log'),
        level=logging.INFO
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    ##############################
    # data_train = NTU_RGBD(root_path = opt.root_path,opt=opt,
    #     DATA_CROSS_VIEW = False,
    #     full_train = True,
    #     validation = False,
    #     test = False,
    #     Transform = True
    #     )
    
    data_train = PointSeriesDataset(
        data_root_dir='/home/appuser/chronopoints_cls_benchmark',
        split='train',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=opt.config
    )
    opt.Num_Class = data_train.num_types
    
    tets = data_train[0]
    
    train_loader = DataLoader(dataset = data_train, batch_size = opt.batchSize, shuffle = True, drop_last = True,num_workers = 8)
    
    
    # data_val = NTU_RGBD(root_path = opt.root_path, opt=opt,
    #     DATA_CROSS_VIEW = False,
    #     full_train = False,
    #     validation = False,
    #     test = True,
    #     Transform = False
    #     )
    data_val = PointSeriesDataset(
        data_root_dir='/home/appuser/chronopoints_cls_benchmark',
        split='val',
        max_points_per_frame=512,
        single_return_only=True,
        preload=True,
        sampling_strategy="farthest",
        padding_strategy="zero_padding",
        sequence_format=opt.config
    )
    val_loader = DataLoader(dataset = data_val, batch_size = 24,num_workers = 8)

    netR = PointNet_Plus(opt)

    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()
    print(netR)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=opt.gamma)
    best_mAcc = 0.0

    for epoch in range(opt.nepoch):
        scheduler.step(epoch)
        
        # switch to train mode
        torch.cuda.synchronize()
        netR.train()
        loss_sigma = 0.0
        conf_mat = np.zeros((opt.Num_Class, opt.Num_Class))
        timer = time.time()
        
        for i, data in enumerate(tqdm(train_loader, 0)):
            if len(data[0])==1:
                continue
            torch.cuda.synchronize()
            # 1 load imputs and target
            ## 3DV points and 3 temporal segment appearance points
            ## points_xyzc: B*2048*8;points_1xyz:B*2048*3  target: B*1
            points4DV_T,label,v_name = data
            points4DV_T,label = points4DV_T.cuda(),label.cuda()
            # print('points4DV_T:',points4DV_T.shape)
            xt, yt = group_points_4DV_T_S(points4DV_T, opt)#B*F*4*Cen*K  B*F*4*Cen*1
            # print('xt:',xt.shape)
            xt = xt.type(torch.FloatTensor)
            yt = yt.type(torch.FloatTensor)

            prediction = netR(xt,yt)

            loss = criterion(prediction,label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            # update training error
            loss_sigma += loss.item()
            #_, predicted60 = torch.max(prediction.data[:,0:60], 1)
            _, predicted = torch.max(prediction.data, 1)
            # print(predicted.data)
            for t, p in zip(label.view(-1), predicted.view(-1)):
                conf_mat[t.item(), p.item()] += 1

        
        OA = conf_mat.trace() / conf_mat.sum()

        class_acc = []
        for i in range(opt.Num_Class):
            if conf_mat[i].sum() == 0:
                class_acc.append(0)
            else:
                class_acc.append(conf_mat[i, i] / conf_mat[i].sum())

        mAcc = np.mean(class_acc)
        loss_avg = loss_sigma / conf_mat.sum()

        print(f"[TRAIN] Epoch {epoch} | OA: {OA:.4f} | mAcc: {mAcc:.4f} | Loss: {loss_avg:.4f}")
        logging.info(f"[TRAIN] Epoch {epoch} | OA: {OA:.4f} | mAcc: {mAcc:.4f} | Loss: {loss_avg:.4f}")
        if ((epoch+1)%1==0 or epoch==opt.nepoch-1):
            torch.cuda.synchronize()
            netR.eval()

            conf_mat = np.zeros((opt.Num_Class, opt.Num_Class))
            loss_sigma = 0.0

            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    points4DV_T, label, v_name = data
                    points4DV_T, label = points4DV_T.cuda(), label.cuda()

                    xt, yt = group_points_4DV_T_S(points4DV_T, opt)
                    xt = xt.type(torch.FloatTensor)
                    yt = yt.type(torch.FloatTensor)

                    prediction = netR(xt, yt)
                    loss = criterion(prediction, label)

                    _, predicted = torch.max(prediction.data, 1)

                    loss_sigma += loss.item()

                    for t, p in zip(label.view(-1), predicted.view(-1)):
                        conf_mat[t.item(), p.item()] += 1

            OA = conf_mat.trace() / conf_mat.sum()

            class_acc = []
            for i in range(opt.Num_Class):
                if conf_mat[i].sum() == 0:
                    class_acc.append(0)
                else:
                    class_acc.append(conf_mat[i, i] / conf_mat[i].sum())

            mAcc = np.mean(class_acc)
            
            print(f"[VAL] Epoch {epoch} | OA: {OA:.4f} | mAcc: {mAcc:.4f} | Loss: {loss_sigma/(i+1):.4f}")
            logging.info(f"[VAL] Epoch {epoch} | OA: {OA:.4f} | mAcc: {mAcc:.4f} | Loss: {loss_sigma/(i+1):.4f}")
            
            if mAcc > best_mAcc:
                best_mAcc = mAcc
                torch.save(netR.module.state_dict(), os.path.join(opt.save_root_dir, 'best_mAcc.pth'))
                print("🔥 New best model saved")
                logging.info("🔥 New best model saved")

            
if __name__ == '__main__':
    main()

