import torch
import numpy as np
import torch.nn as nn


_EPSILON = 1e-10

def soft_dice(true, pred, eps: float = 1e-6):
    """Compute soft dice on a batch of predictions.

    The soft dice is defined as follows with N denoting all pixels in an image.
    According to [Wollmann, 2019](https://ieeexplore.ieee.org/abstract/document/8759234/),
    calculating the Dice loss over all N pixels in a batch instead of averaging the Dice
    loss over the single images improves training stability.
    """

    # [b, h, w, 1] -> [b*h*w*1]
    true = torch.flatten(true)
    pred = torch.flatten(pred)

    # [sum(b), h*w]
    multed = torch.sum(true * pred)
    summed = torch.sum(true + pred)
    dices = 2.0 * ((multed + eps) / (summed + eps))

    return -dices


class Soft_dice(nn.Module):
    def __init__(self) -> None:
        super(Soft_dice,self).__init__()
    def forward(self,pred,true):
        return soft_dice(true,pred)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        torch.save({'model_state_dict': model.state_dict(),
                    'loss': val_loss},
                    self.path)


        self.val_loss_min = val_loss



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
    precision = precision_score(y_true[:,0,:,:], y_pred[:,0,:,:])
    recall = recall_score(y_true[:,0,:,:], y_pred[:,0,:,:])
    f1_value = 2 * ((precision * recall) / (precision + recall + _EPSILON))
    return f1_value