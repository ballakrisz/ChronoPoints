import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
from pointnet2_ops import pointnet2_utils

def group_points_4DV_T_S(points, opt):
    #B*F*512*4
    T_ball_radius = torch.tensor(0.06)
    cur_train_size = points.shape[0]#
    INPUT_FEATURE_NUM = points.shape[-1]#3
    
    points = points.view(cur_train_size*opt.framenum, opt.EACH_FRAME_SAMPLE_NUM, -1)#(B*F)*512*4
    # print('1points:',points.shape,cur_train_size,opt.framenum, opt.EACH_FRAME_SAMPLE_NUM, -1)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size*opt.framenum,opt.T_sample_num_level1,3,opt.EACH_FRAME_SAMPLE_NUM) \
                 - points[:,0:opt.T_sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size*opt.framenum,opt.T_sample_num_level1,3,opt.EACH_FRAME_SAMPLE_NUM)# (B*F )* 64 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024 distance
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.T_knn_K, 2, largest=False, sorted=False)  # dists: B * 512 * 32; inputs1_idx: B * 512 * 32
    
    # ball query
    invalid_map = dists.gt(T_ball_radius) # B * 512 * 64  value: binary
    
    for jj in range(opt.T_sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size*opt.framenum,opt.T_sample_num_level1*opt.T_knn_K,1).expand(cur_train_size*opt.framenum,opt.T_sample_num_level1*opt.T_knn_K,INPUT_FEATURE_NUM)

    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size*opt.framenum,opt.T_sample_num_level1,opt.T_knn_K,INPUT_FEATURE_NUM) # (B*F)*64*32*4

    inputs_level1_center = points[:,0:opt.T_sample_num_level1,0:INPUT_FEATURE_NUM ].unsqueeze(2)       # (B*F)*64*1*4
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center[:,:,:,0:3].expand(cur_train_size*opt.framenum,opt.T_sample_num_level1,opt.T_knn_K,3)# (B*F)*64*32*3
    if(1==1):
        dis_l=torch.mul(inputs_level1[:,:,:,0:3], inputs_level1[:,:,:,0:3])
        
        dis_l=dis_l.sum(3).unsqueeze(3)#lx#
        
        inputs_level1 = torch.cat((inputs_level1,dis_l),3).unsqueeze(1).transpose(1,4).squeeze(4)  # (B*F)*4*64*32
    
    inputs_level1_center = inputs_level1_center.contiguous().view(cur_train_size,opt.framenum,opt.T_sample_num_level1,1,INPUT_FEATURE_NUM).transpose(2,3).transpose(2,4)   # (B*F)*4*64*1
    FEATURE_NUM = inputs_level1.shape[-3]#4
    inputs_level1=inputs_level1.view(cur_train_size,opt.framenum,FEATURE_NUM, opt.T_sample_num_level1, opt.T_knn_K)#B*F*4*Cen*K
    return inputs_level1, inputs_level1_center
def group_points_4DV_T_S2(points, opt):

    T_ball_radius = 0.11

    B = points.shape[0]
    F = opt.framenum
    C = points.shape[-1]

    BF = B * F

    # =====================================================
    # Flatten temporal dimension
    # =====================================================

    points = points.view(
        BF,
        opt.T_sample_num_level1,
        C
    ).contiguous()

    xyz = points[:, :, 0:3].contiguous()

    # =====================================================
    # Centers
    # =====================================================

    center_xyz = xyz[:, 0:opt.T_sample_num_level2, :].contiguous()

    # =====================================================
    # Pairwise distances
    # =====================================================

    dists = torch.cdist(
        center_xyz,
        xyz
    ) ** 2

    # =====================================================
    # KNN
    # =====================================================

    dists, idx = torch.topk(
        dists,
        opt.T_knn_K2,
        dim=2,
        largest=False,
        sorted=False
    )

    # =====================================================
    # Radius mask
    # =====================================================

    invalid_map = dists > T_ball_radius

    # =====================================================
    # Replace invalid neighbors with center idx
    # =====================================================

    for jj in range(opt.T_sample_num_level2):

        idx[:, jj, :][invalid_map[:, jj, :]] = jj

    idx = idx.int().contiguous()

    # =====================================================
    # CUDA grouping
    # =====================================================

    grouped = pointnet2_utils.grouping_operation(
        points.transpose(1, 2).contiguous(),
        idx
    )

    grouped = grouped.permute(
        0,
        2,
        3,
        1
    ).contiguous()

    # =====================================================
    # Centers
    # =====================================================

    inputs_level1_center = points[
        :,
        0:opt.T_sample_num_level2,
        0:opt.INPUT_FEATURE_NUM
    ].unsqueeze(2)

    # =====================================================
    # Relative xyz
    # =====================================================

    grouped[:, :, :, 0:3] = (
        grouped[:, :, :, 0:3]
        - inputs_level1_center[:, :, :, 0:3]
    )

    # =====================================================
    # Distance feature
    # =====================================================

    dis_l = torch.sum(
        grouped[:, :, :, 0:3] ** 2,
        dim=3,
        keepdim=True
    )

    grouped = torch.cat(
        [grouped, dis_l],
        dim=3
    )

    # =====================================================
    # Match original layout
    # =====================================================

    grouped = (
        grouped
        .unsqueeze(1)
        .transpose(1, 4)
        .squeeze(4)
        .contiguous()
    )

    FEATURE_NUM = grouped.shape[-3]

    grouped = grouped.view(
        B,
        F,
        FEATURE_NUM,
        opt.T_sample_num_level2,
        opt.T_knn_K2
    )

    inputs_level1_center = (
        inputs_level1_center
        .contiguous()
        .view(
            B,
            F,
            opt.T_sample_num_level2,
            1,
            opt.INPUT_FEATURE_NUM
        )
        .transpose(2, 3)
        .transpose(2, 4)
        .contiguous()
    )

    return grouped, inputs_level1_center