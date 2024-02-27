# from turtle import forward
# import tensorflow as tf
# import tensorflow.keras.backend as K
import torch.nn as nn

import torch

_EPSILON = 1e-10

def dice_score(y_true, y_pred, smooth: int = 1):
    r"""Computes the dice coefficient on a batch of tensors.

    .. math::
        \textrm{Dice} = \frac{2 * {\lvert X \cup Y\rvert}}{\lvert X\rvert +\lvert Y\rvert}


    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    Args:
        y_true: Ground truth masks.
        y_pred: Predicted masks.
        smooth: Epslion value to avoid division by zero.
    """

    y_true_f = torch.flatten(y_true)                
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    """Dice score loss corresponding to deepblink.losses.dice_score."""
    return 1 - dice_score(y_true, y_pred)

def rmse(y_true, y_pred): #只想计算前景的坐标预测情况
    """Calculate root mean square error (rmse) between true and predicted coordinates."""
    # RMSE, takes in the full y_true/y_pred when used as metric.
    # Therefore, do not move the selection outside the function.
    # y_true = y_true[..., 1:]
    # y_pred = y_pred[..., 1:]

    comparison = (y_true == 0) #制作应该为0的mask,背景的mask

    y_true_new = torch.where(comparison, torch.zeros_like(y_true), y_true)#不变
    y_pred_new = torch.where(comparison, torch.zeros_like(y_pred), y_pred)#将所有背景置为0，

    sum_rc_coords = torch.sum(y_true, axis=1) #x y 偏移量之和
    n_true_spots = torch.count_nonzero(sum_rc_coords) #非0数量，前景的数量

    squared_displacement_xy_summed = torch.sum(torch.square(y_true_new - y_pred_new), axis=1) #计算的是预测的前景的坐标差的平方和的和，其他的也都是0
    rmse_value = torch.sqrt(
        torch.sum(squared_displacement_xy_summed) / (n_true_spots + _EPSILON)
    )

    return rmse_value

def combined_dice_rmse(y_true, y_pred):
    """Loss that combines dice for probability and rmse for coordinates.

    The optimal values for dice and rmse are both 0.
    """
    return dice_loss(y_true[:,0,:,:], y_pred[..., 0]) + rmse(y_true[:,1:,:,:], y_pred[...,1:].transpose(0,3,1,2)) * 2


class Combined_dice_rmse(nn.Module):
    def __init__(self) -> None:
        super(Combined_dice_rmse,self).__init__()

    def forward(self,y_pred,y_true):
        return dice_loss(y_true[:,:,:,0], y_pred[:,0,:,:]) + rmse(y_true[:,:,:,1:].permute(0,3,1,2), y_pred[:,1:,:,:]) * 2





def recall_score(y_true, y_pred):
    """Recall score metric.

    Defined as ``tp / (tp + fn)`` where tp is the number of true positives and fn the number of false negatives.
    Can be interpreted as the accuracy of finding positive samples or how many relevant samples were selected.
    The best value is 1 and the worst value is 0.
    """
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + _EPSILON)
    return recall


def precision_score(y_true, y_pred):
    """Precision score metric.

    Defined as ``tp / (tp + fp)`` where tp is the number of true positives and fp the number of false positives.
    Can be interpreted as the accuracy to not mislabel samples or how many selected items are relevant.
    The best value is 1 and the worst value is 0.
    """
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + _EPSILON)
    return precision


def f1_score(y_pred,y_true):
    r"""F1 score metric.

    .. math::
        F1 = \frac{2 * \textrm{precision} * \textrm{recall}}{\textrm{precision} + \textrm{recall}}

    The equally weighted average of precision and recall.
    The best value is 1 and the worst value is 0.
    """
    # Do not move outside of function. See RMSE.
    precision = precision_score(y_true[:,:,:,0], y_pred[:,0,:,:])
    recall = recall_score(y_true[:,:,:,0], y_pred[:,0,:,:])
    f1_value = 2 * ((precision * recall) / (precision + recall + _EPSILON))
    return f1_value