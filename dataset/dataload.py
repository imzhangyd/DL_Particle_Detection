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
        from dataset.superpoint_Dataset import cls_Dataset
    elif model_mode == 'DetNet':
        from dataset.DetNet_Dataset import cls_Dataset
    elif model_mode == 'deepBlink':
        from dataset.deepBlink_Dataset import cls_Dataset_16
    dtst_ins = cls_Dataset_16(txtfile, training=training)
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins

# 16 bit的情况只有deepBlink有实现，其他的补齐

# for only pred
def func_getdataloader_pred(model_mode, txtfile, batch_size, shuffle, num_workers, training=True):
    if model_mode == 'superpoint':
        from dataset.superpoint_Dataset import cls_Dataset
        dtst_ins = cls_Dataset(txtfile)
    elif model_mode == 'DetNet':
        from dataset.DetNet_Dataset import cls_Dataset_onlypred
        dtst_ins = cls_Dataset_onlypred(txtfile,training=training)
    elif model_mode == 'deepBlink':
        from dataset.deepBlink_Dataset import cls_Dataset_onlypred
    
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins


def func_getdataloader_pred_16(model_mode, txtfile, batch_size, shuffle, num_workers, training=True):
    if model_mode == 'superpoint':
        from dataset.superpoint_Dataset import cls_Dataset
        dtst_ins = cls_Dataset(txtfile)
    elif model_mode == 'DetNet':
        from dataset.DetNet_Dataset import cls_Dataset_onlypred
        dtst_ins = cls_Dataset_onlypred(txtfile)
    elif model_mode == 'deepBlink':
        from dataset.deepBlink_Dataset import cls_Dataset_onlypred_16
        dtst_ins = cls_Dataset_onlypred_16(txtfile,training=training)

    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins
