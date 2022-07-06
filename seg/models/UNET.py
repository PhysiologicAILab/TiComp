
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=[1, 1], padding=0, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_channels, n_classes, n_features=24, kernel=3, use_dropout=False, seg_net_out_opt=1):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_filters = self.n_features * np.array([1, 2, 4, 8, 16])
        self.use_dropout = use_dropout
        self.seg_net_out_opt = seg_net_out_opt
        
        self.conv1 = ConvBlock(self.n_channels, self.n_filters[0], kernel_size=kernel, stride=[1, 2], padding=1, use_dropout=False)
        self.conv2 = ConvBlock(self.n_filters[0], self.n_filters[1], kernel_size=kernel, stride=[1, 2], padding=1, use_dropout=False)
        self.conv3 = ConvBlock(self.n_filters[1], self.n_filters[2], kernel_size=kernel, stride=[1, 2], padding=1, use_dropout=False)
        self.conv4 = ConvBlock(self.n_filters[2], self.n_filters[3], kernel_size=kernel, stride=[1, 2], padding=1, use_dropout=False)
        self.conv5 = ConvBlock(self.n_filters[3], self.n_filters[3], kernel_size=kernel, stride=[1, 2], padding=1, use_dropout=False)
        self.conv6 = ConvBlock(self.n_filters[4], self.n_filters[2], kernel_size=kernel, stride=[1, 1], padding=1, use_dropout=self.use_dropout)
        self.conv7 = ConvBlock(self.n_filters[3], self.n_filters[1], kernel_size=kernel, stride=[1, 1], padding=1, use_dropout=self.use_dropout)
        self.conv8 = ConvBlock(self.n_filters[2], self.n_filters[0], kernel_size=kernel, stride=[1, 1], padding=1, use_dropout=self.use_dropout)
        self.conv9 = ConvBlock(self.n_filters[1], self.n_filters[0], kernel_size=kernel, stride=[1, 1], padding=1, use_dropout=False)
        self.conv10 = nn.Conv2d(self.n_filters[0], n_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):

        x1 = self.conv1(x)

        x2 = self.conv2(x1)

        x3 = self.conv3(x2)

        x4 = self.conv4(x3)

        x5 = self.conv5(x4)

        x6 = nn.functional.interpolate(
            x5, x4.size()[2:], mode='bilinear', align_corners=False)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.conv6(x6)

        x7 = nn.functional.interpolate(
            x6, x3.size()[2:], mode='bilinear', align_corners=False)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.conv7(x7)

        x8 = nn.functional.interpolate(
            x7, x2.size()[2:], mode='bilinear', align_corners=False)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.conv8(x8)

        x9 = nn.functional.interpolate(
            x8, x1.size()[2:], mode='bilinear', align_corners=False)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.conv9(x9)

        x10 = nn.functional.interpolate(
            x9, x.size()[2:], mode='bilinear', align_corners=False)
        out = self.conv10(x10)

        if self.seg_net_out_opt == 1:
            return out
        else:
            return out, x10
