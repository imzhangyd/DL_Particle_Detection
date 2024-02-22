from torch.utils.data import DataLoader
from dataset.custom_Dataset import cls_Dataset


def func_getdataloader(txtfile, batch_size, shuffle, num_workers):
    dtst_ins = cls_Dataset(txtfile)
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins
    