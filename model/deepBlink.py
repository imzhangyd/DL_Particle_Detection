import torch.nn.functional as F
import torch
import torch.nn as nn
import glob
import os
import numpy as np
import cv2

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1,1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        cat_x = torch.cat((x1, x2), 1)
        output = self.conv(cat_x)
        return output



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



'''
model deepBlink
'''
class deepBlink(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(deepBlink, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ###  v1 me
        # # encoder
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)

        # # decoder
        # self.up3 = Up(256, 128, bilinear)
        # self.up4 = Up(128, 64, bilinear)

        # # encoder2
        # self.down1_ = Down(64, 128)
        # self.down2_ = Down(128, 256)

        # self.dropout = nn.Dropout2d(p=0.2)
        # self.maxpool = nn.MaxPool2d(2)

        # self.conv_ = DoubleConv(256,256)
        # self.conv__ = nn.Conv2d(256, 3, kernel_size=1)

        ###  v2 code
        # encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)

        # decoder
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)

        # encoder2
        self.down1_ = Down(32, 64)
        self.down2_ = Down(64, 128)

        self.dropout = nn.Dropout2d(p=0.3)
        self.maxpool = nn.MaxPool2d(2)

        self.conv_ = DoubleConv(128,128)
        self.conv__ = nn.Conv2d(128, 3, kernel_size=1)


    def forward(self, x):
        # encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)


        # decoder
        # o_4 = self.up1(x5, x4)
        # o_3 = self.up2(o_4, x3)
        o_2 = self.up3(x3, self.dropout(x2))
        o_1 = self.up4(o_2, self.dropout(x1))

        e_1 = self.down1_(o_1)
        e_2 = self.down2_(o_2)

        y = torch.cat((self.maxpool(x2),self.maxpool(e_1)), 1)
        y_ = self.conv_(y)
        y__ = self.conv__(y_)

        # o_seg = self.out(o_1)

        seg = torch.sigmoid(y__)

        # if self.n_classes > 1:
        #     seg = F.softmax(y__, dim=1)
        #     return seg
        # elif self.n_classes == 1 :
        #     seg = torch.sigmoid(y__)
        #     return seg
        return seg
