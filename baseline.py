'''
Author: my_zhao
Date: 2022-07-01 20:49:02
LastEditors: my_zhao
LastEditTime: 2023-01-05 18:56:45
Description: 请填写简介
'''
import torch
import torch.nn as nn

import numpy as np
from torch.nn.parallel.data_parallel import DataParallel
from model.UNet import UNet3D
from model.wingnet import WingsNet
from model.Discriminator import Discriminator as D
from option import parser
from model.Thre_regression import PRM_predictor
# Baseline
from monai.networks.nets import UNet, AttentionUnet
global args
args = parser.parse_args()
config = {'pad_value': 0,
          'augtype': {'flip': True, 'swap': True, 'smooth': False, 'jitter': True, 'split_jitter': True},
          'startepoch': 1, 'lr_stage': np.array([10, 20, 40, 50, 70, 80,  100, 200, 400, 800]), 'lr': np.array([0.01, 0.01, 0.001, 0.001,0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]),
          'lr_stage_D': np.array([10, 20, 40, 50, 70, 80,  100, 200, 400, 800]), 'lr_D': np.array([0.01, 0.01, 0.001, 0.001, 0.001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]),
          'dataset_path': 'processed2_data', 'dataset_split': args.dataset_split,
          "deploy_path":"/data/mingyuezhao/PRM/data/External_verification_25_1_7/Old_manCOPD_PRISm_processed",
          'DS_dataset_path': './data/processed2_data_fine_downsam',
          "train_set": "./split_comber_train.pickle",
          'val_set': "./split_comber_val.pickle",
          'test_set': "./split_comber_test.pickle",
          'cube_savepath': "./data/preload_data"}


def get_model():
    # net = PRM_predictor(in_channels=1, out_channels=3)
    net = UNet3D(in_channels=1, out_channels=1)
    # D_net = D(in_ch=1, out_ch=1)
    # net = UNet(
    # spatial_dims=3,
    # in_channels=1,
    # out_channels=3,
    # channels=(8, 16,32,64,128),
    # strides=(2, 2,2,2),
    # num_res_units=4
    # )
    # net = AttentionUnet(
    #     spatial_dims=3,
    #     in_channels=2,
    #     out_channels=3,
    #     channels=(8, 16, 32, 64, 128),
    #     strides=(2, 2, 2, 2)
    # )
    # net = INR_Seg(hidden_dim=256)
    # net = WingsNet(in_channel=2, n_classes=3)
    # print(net)
    # print('# of network parameters:', sum(param.numel()
    #       for param in net.parameters()))
    # return config, G_net,D_net
    return config, net


if __name__ == '__main__':
    _, model = get_model()
    # load_pickle()
