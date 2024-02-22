from loss.iou import cls_IOUloss
from torch.nn import CrossEntropyLoss,BCELoss
from loss.deepblinkloss import Combined_dice_rmse
from loss.detnetloss import Soft_dice

def func_getloss(name = 'iou',classnum = 1):
    if name == 'iou':
        return cls_IOUloss()
    elif name == 'crossentropy':
        if classnum == 1:
            return BCELoss()
        else:
            return CrossEntropyLoss()
    elif name == 'combined_dice_rmse':
        return Combined_dice_rmse()
    elif name == 'soft_dice':
        return Soft_dice()
    


