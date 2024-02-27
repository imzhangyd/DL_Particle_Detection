import torch.nn as nn
import torch
from torch import sigmoid


__author__ = "Yudong Zhang"


class cls_conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(cls_conv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), # padding=1 considering size not change
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.double_conv(x)
        return x

class cls_up(nn.Module):
    def __init__(self,in_channels, out_channels) -> None:
        super(cls_up,self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride = 2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv22 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), # kernel_size is right
        
    def forward(self,x):
        # y = self.conv22(self.upsample(x))
        y = self.upsample(x)
        return y



class cls_Unet(nn.Module):
    def __init__(self,imgchannel,num_class): #num_class is beforeground class number+1
        super(cls_Unet, self).__init__()
        self.num_class = num_class
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.d1 = cls_conv(imgchannel,64)
        self.d2 = cls_conv(64,128)
        self.d3 = cls_conv(128,256)
        self.d4 = cls_conv(256,512)
        self.d5 = cls_conv(512,1024)

        self.u4 = cls_up(1024,512)
        self.u4_ = cls_conv(1024,512)
        self.u3 = cls_up(512,256)
        self.u3_ = cls_conv(512,256)
        self.u2 = cls_up(256,128)
        self.u2_ = cls_conv(256,128)
        self.u1 = cls_up(128,64)
        self.u1_ = cls_conv(128,64)

        self.out_1 = nn.Conv2d(64,1, kernel_size=1, stride=1, padding=0)
        self.out_2 = nn.Conv2d(64,num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        it1 = self.d1(x)
        it2 = self.d2(self.downsample(it1))
        it3 = self.d3(self.downsample(it2))
        it4 = self.d4(self.downsample(it3))
        it5 = self.d5(self.downsample(it4))


        ot4 = self.u4_(torch.cat((self.u4(it5), it4), 1))
        ot3 = self.u3_(torch.cat((self.u3(ot4),it3),1))
        ot2 = self.u2_(torch.cat((self.u2(ot3),it2),1))
        ot1 = self.u1_(torch.cat((self.u1(ot2),it1),1))

        if self.num_class == 1:
            y = sigmoid(self.out_1(ot1))
        else:
            y = self.out_2(ot1)

        return [y]


if __name__ == '__main__':
    model = cls_Unet(1,1)
    print(model.parameters())
    print("==> List learnable parameters")
    print(model.named_parameters())
    for name, param in model.named_parameters():
        print("\t{}, size {}".format(name, param.size()))
        if param.requires_grad == True:
            print("\t{}, size {}".format(name, param.size()))
    print(model)
    # params_to_update = [{'params': model.parameters()}]