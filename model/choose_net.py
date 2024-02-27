from model.unet import cls_Unet
from model.unetunet import cls_UnetUnet
from model.old_unet import UNet,UnetUnet
from model.deepBlink import deepBlink
from model.DetNet import cls_DetNet
from torch.nn.functional import alpha_dropout
from model.superPoint import SuperPointNet
from model.hrnet_dekr import PoseHigherResolutionNet


def func_getnetwork(name,opt):
    if name == 'unet':
        # return cls_Unet(1,1)
        return UNet(1)
    elif name == 'unetunet':
        # return cls_UnetUnet(1,1)
        return(UnetUnet(1))
    elif name == 'deepBlink':
        return(deepBlink(1))
    elif name == 'DetNet':
        return(cls_DetNet(alpha = opt['alpha']))
    elif name == 'superpoint':
        return(SuperPointNet())
    elif name == 'PointDet':
        from config.default import _C as cfg
        from config.default import update_config
        update_config(cfg, opt)
        return(PoseHigherResolutionNet(cfg))

