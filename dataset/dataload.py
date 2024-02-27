from torch.utils.data import DataLoader



def func_getdataloader(model_mode, txtfile, batch_size, shuffle, num_workers,training=True):
    if model_mode == 'superpoint':
        from dataset.superpoint_Dataset import cls_Dataset
    elif model_mode == 'DetNet':
        from dataset.DetNet_Dataset import cls_Dataset
    elif model_mode == 'deepBlink':
        from dataset.deepBlink_Dataset import cls_Dataset
    elif model_mode == 'PointDet':
        from dataset.PointDet_Dataset import cls_Dataset
        
    dtst_ins = cls_Dataset(txtfile, training=training)
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins

def func_getdataloader_16(model_mode, txtfile, batch_size, shuffle, num_workers,training=True):
    if model_mode == 'superpoint':
        from dataset.superpoint_Dataset import cls_Dataset_16
    elif model_mode == 'DetNet':
        from dataset.DetNet_Dataset import cls_Dataset_16
    elif model_mode == 'deepBlink':
        from dataset.deepBlink_Dataset import cls_Dataset_16
    elif model_mode == 'PointDet':
        from dataset.PointDet_Dataset import cls_Dataset_16
        
    dtst_ins = cls_Dataset_16(txtfile, training=training)
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins


# for only predicting without label to evaluate
def func_getdataloader_pred(model_mode, txtfile, batch_size, shuffle, num_workers, training=False):
    if model_mode == 'superpoint':
        from dataset.superpoint_Dataset import cls_Dataset_onlypred
        dtst_ins = cls_Dataset_onlypred(txtfile)
    elif model_mode == 'DetNet':
        from dataset.DetNet_Dataset import cls_Dataset_onlypred
        dtst_ins = cls_Dataset_onlypred(txtfile)
    elif model_mode == 'deepBlink':
        from dataset.deepBlink_Dataset import cls_Dataset_onlypred
        dtst_ins = cls_Dataset_onlypred(txtfile)
    else:
        print('Error! we have not finish this dataset class for {}'.format(model_mode))
            
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins


def func_getdataloader_pred_16(model_mode, txtfile, batch_size, shuffle, num_workers, training=False):
    if model_mode == 'superpoint':
        from dataset.superpoint_Dataset import cls_Dataset_onlypred_16
        dtst_ins = cls_Dataset_onlypred_16(txtfile)
    elif model_mode == 'DetNet':
        from dataset.DetNet_Dataset import cls_Dataset_onlypred_16
        dtst_ins = cls_Dataset_onlypred_16(txtfile)
    elif model_mode == 'deepBlink':
        from dataset.deepBlink_Dataset import cls_Dataset_onlypred_16
        dtst_ins = cls_Dataset_onlypred_16(txtfile)
    else:
        print('Error! we have not finish this dataset class for {}'.format(model_mode))

    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins
