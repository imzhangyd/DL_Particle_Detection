import torch.nn as nn
from torch.nn import Linear
from model.unet import cls_Unet
class cls_UnetUnet(nn.Module):
    def __init__(self,imgchannel,num_class):
        super(cls_UnetUnet, self).__init__()
        self.unet1 = cls_Unet(imgchannel,num_class)
        self.unet2 = cls_Unet(imgchannel,num_class)

    def forward(self, x):
        y1 = self.unet1(x)
        y2 = self.unet1(y1[0]*0.5+x*0.5)
        return [y1[0],y2[0]]