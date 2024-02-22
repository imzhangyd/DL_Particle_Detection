
def func_ioucreterion(pred,label,threshold):
    b_pred = pred > threshold
    ioulist = []
    for _ in range(pred.shape[0]):
        i = ((b_pred[_,0,:,:]*label[_,0,:,:])>0.5).flatten().sum()
        u = ((b_pred[_,0,:,:]+label[_,0,:,:])>0.5).flatten().sum()
        iou = i/u
        ioulist.append(iou)
    return ioulist
