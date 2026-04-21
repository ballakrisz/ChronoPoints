import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from intra_mamba import *
from mamba import *
from utils_mamba import Group, Encoder
from point_4d_convolution import *

class MAMBA4D(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, mlp_dim, num_classes,                                                 # output
                 depth_mamba_inter, rms_norm, 
                 drop_out_in_block, drop_path,
                 depth_mamba_intra, intra):
        super().__init__()

        feature_extraction_params = {
            'in_planes': 0,
            'mlp_planes': [dim],
            'mlp_batch_norm': [False],
            'mlp_activation': [False],
            'spatial_kernel_size': [radius, nsamples],
            'spatial_stride': spatial_stride,
            'temporal_kernel_size': temporal_kernel_size,
            'temporal_stride': temporal_stride,
            'temporal_padding': [1, 0],
            'operator': '+',
            'spatial_pooling': 'max',
            'temporal_pooling': 'max',
            'depth_mamba_intra': depth_mamba_intra
        }
        
        self.tube_embedding = IntraMamba(**feature_extraction_params) if intra else P4DConv(**feature_extraction_params)
            

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.mambaBlocks = MixerModel(d_model=dim,
                            n_layer=depth_mamba_inter,
                            rms_norm=rms_norm,
                            drop_out_in_block=drop_out_in_block,
                            drop_path=drop_path)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        # sort
        x_labels = []
        y_labels = []
        z_labels = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
            # sort
            xyz_index = torch.argsort(xyzt, dim=1)
            x_labels.append(xyz_index[:, :, 0].unsqueeze(-1))
            y_labels.append(xyz_index[:, :, 1].unsqueeze(-1))
            z_labels.append(xyz_index[:, :, 2].unsqueeze(-1))

        # sort
        x_labels, y_labels, z_labels = torch.cat(x_labels, dim=1), torch.cat(y_labels, dim=1), torch.cat(z_labels, dim=1)
        x_labels = torch.argsort(x_labels, dim=1, stable=True)
        y_labels = torch.argsort(y_labels, dim=1, stable=True)
        z_labels = torch.argsort(z_labels, dim=1, stable=True)

        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        # sort
        embedding_x = torch.gather(embedding, 1, x_labels.expand(-1, -1, embedding.size(-1)))
        embedding_y = torch.gather(embedding, 1, y_labels.expand(-1, -1, embedding.size(-1)))
        embedding_z = torch.gather(embedding, 1, z_labels.expand(-1, -1, embedding.size(-1)))

        embedding = torch.cat([embedding_z, embedding_x, embedding_y], dim=1)

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.mambaBlocks(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output
