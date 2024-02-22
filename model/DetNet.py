
import torch.nn as nn

class conv_norm(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size = 3,padding = 1):
        super().__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,padding=padding),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(num_features=out_channel)
        )

    def forward(self,x):
        x = self.conv_norm(x)
        return x


class Down(nn.Module):
    def __init__(self,inchannel,outchannel) -> None:
        super().__init__()

        self.conv_norm1 = conv_norm(inchannel,outchannel)
        self.conv_norm2 = conv_norm(outchannel,outchannel)
        self.conv_norm3 = conv_norm(outchannel,outchannel)

        self.maxpooling1 = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        conv1 = self.conv_norm1(x)
        skip1 = self.conv_norm2(conv1)
        skip2 = self.conv_norm3(skip1)
        add1 = conv1+skip2
        max1 = self.maxpooling1(add1)
        return max1


class Up(nn.Module):
    def __init__(self,inchannel,outchannel) -> None:
        super(Up,self).__init__()

        self.conv_norm1 = conv_norm(inchannel,outchannel)
        self.conv_norm2 = conv_norm(outchannel,outchannel)
        self.conv_norm3 = conv_norm(outchannel,outchannel)

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,x):
        conv1 = self.conv_norm1(x)
        skip1 = self.conv_norm2(conv1)
        skip2 = self.conv_norm3(skip1)
        add1 = conv1+skip2
        max1 = self.upsampling(add1)
        return max1


class Out(nn.Module):
    def __init__(self,inchannel,outchannel,alpha) -> None:
        super().__init__()
        self.conv_norm1 = conv_norm(inchannel,outchannel)
        self.conv_norm2 = conv_norm(outchannel,outchannel)
        self.conv_norm3 = conv_norm(outchannel,outchannel)

        self.conv_norm4 = nn.Conv2d(outchannel,1,kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha

    def forward(self,x):
        conv1 = self.conv_norm1(x)
        skip1 = self.conv_norm2(conv1)
        skip2 = self.conv_norm3(skip1)
        add1 = conv1+skip2

        out = self.conv_norm4(add1)
        out = self.sigmoid(out-self.alpha)

        return out




class cls_DetNet(nn.Module):
    
    """Return the DetNet architecture.

    Args:
        image_size: Size of input images with format (image_size, image_size).
        alpha: Balance parameter in the sigmoid layer.
    """

    def __init__(self, alpha):
        super(cls_DetNet, self).__init__()

        self.down1 = Down(1,16)
        self.down2 = Down(16,32)

        self.up1 = Up(32,64)
        self.up2 = Up(64,32)

        self.out = Out(32,16,alpha)
        
    def forward(self, x):
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.out(x)

        return x

        