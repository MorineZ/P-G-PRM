from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import DataParallel
# from torchkeras import summary


class P_PRM(nn.Module):
    """
    Baseline model for pulmonary airway segmentation
    """

    def __init__(self, in_channels=1, out_channels=1):
        """
        :param in_channels: input channel numbers
        :param out_channels: output channel numbers
        :param coord: boolean, True=Use coordinates as position information, False=not
        """
        super(P_PRM, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.conv1: nn.Module
        # self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # modfied by mingyuezhao
        self.pooling1 = nn.Conv3d(16, 16,
                      kernel_size=3, stride=2,padding=1)
        self.pooling2 = nn.Conv3d(32, 32,
                      kernel_size=3, stride=2,padding=1)
        self.pooling3 = nn.Conv3d(64, 64,
                      kernel_size=3, stride=2,padding=1)
        self.pooling4 = nn.Conv3d(128, 128,
                      kernel_size=3, stride=2,padding=1)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(self._in_channels, 8,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True))
        # nn.Tanh())

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv3d(16, 32, 3, 1, 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True))
        # nn.Tanh())

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))
        # nn.Tanh())

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True))
        # nn.Tanh())

        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv3d(128, 256, 3, 1, 1),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True))
        # nn.Tanh())
       # 扩张路径
        self.conv6 = nn.Sequential(
            nn.Conv3d(256 + 128, 128, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True))
        # nn.Tanh())

        self.conv7 = nn.Sequential(
            nn.Conv3d(128 + 64, 64, 1, 1, padding=0),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))
        # nn.Tanh())

        self.conv8 = nn.Sequential(
            nn.Conv3d(64 + 32, 32, 1, 1, padding=0),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True))
        # nn.Tanh())

        self.conv9 = nn.Sequential(
            nn.Conv3d(32 + 16, 16, 1, 1, padding=0),
            nn.InstanceNorm3d(16),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            # nn.Tanh())
            nn.ReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()
        self.conv10 = nn.Conv3d(16, self._out_channels, 1, 1, 0)

        # data_parallel

    def forward(self, input):
        """
        :param input: shape = (batch_size, num_channels, D, H, W) \
        :param coordmap: shape = (batch_size, 3, D, H, W)
        :return: output segmentation tensor, attention mapping
        """

        conv1 = self.conv1(input) 
        x = self.pooling1(conv1)
        conv2 = self.conv2(x)
        x = self.pooling2(conv2)
        conv3 = self.conv3(x)
        x = self.pooling3(conv3)
        conv4 = self.conv4(x)
        x = self.pooling4(conv4)

    # bottleneck
        conv5 = self.conv5(x)
        # upsampling
        x = self.upsampling(conv5)
        x = torch.cat([x, conv4], dim=1)
        conv6 = self.conv6(x)

        x = self.upsampling(conv6)
        x = torch.cat([x, conv3], dim=1)
        conv7 = self.conv7(x)

        x = self.upsampling(conv7)
        x = torch.cat([x, conv2], dim=1)
        conv8 = self.conv8(x)

        x = self.upsampling(conv8)
        x = torch.cat([x, conv1], dim=1)

        conv9 = self.conv9(x)
        x = self.conv10(conv9)
        x = self.sigmoid(x)

        return x
        # return x, nn.ParameterList([nn.Parameter(mapping3),nn.Parameter(mapping4),\
        # 			nn.Parameter(mapping5),nn.Parameter( mapping6 ),nn.Parameter(mapping7),\
        # 				nn.Parameter(mapping8), nn.Parameter(mapping9)])


if __name__ == '__main__':
    net = P_PRM(in_channels=1, out_channels=3)
    # print(net)
    input = torch.randn(1, 1,80, 80,80)
    print(input.shape)
    out = net(input)
    print(out.shape)
    print('Number of network parameters:', sum(param.numel()
          for param in net.parameters()))
