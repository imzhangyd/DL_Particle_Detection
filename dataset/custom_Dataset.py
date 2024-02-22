from numpy import core
from torch.utils.data import Dataset
import cv2
import torch
from dataset.dataprocess import func_normlize
import glob
import numpy as np
import pandas as pd

from utils.data import get_prediction_matrix

def get_seg_maps(arr: np.ndarray, size: int = 512):
    """Convert coordinate list into single-pixel segmentation maps."""

    # # 1pixel
    # seg_maps = np.zeros((1, size, size))
    # for coord in np.round(arr).astype(int):
    #     seg_maps[0,min(coord[1],size -1 ), min (coord[0],size-1)] = 1
    # return seg_maps

    # 4pixel
    seg_maps = np.zeros((1, size, size))
    for coord in np.trunc(arr).astype(int):
        seg_maps[0,min(coord[1],size -1 ), min (coord[0],size-1)] = 1
        seg_maps[0,min(coord[1]+1,size -1), min(coord[0]+1,size-1)] = 1
        seg_maps[0,min(coord[1]+1,size -1), min(coord[0],size-1)] = 1
        seg_maps[0,min(coord[1],size -1), min(coord[0]+1,size-1)] = 1
    return seg_maps

def get_fold_label(arr, size, fold_time=4):
    Hc = int(size / fold_time)
    Wc = int(size / fold_time)
    arr = np.reshape(arr, [Hc, fold_time, Wc, fold_time])
    arr = np.transpose(arr, [0,2,1,3])
    arr = np.reshape(arr, [Hc, Wc, fold_time*fold_time])
    arr = np.transpose(arr,[2,0,1])
    
    # add dustbin channel
    dustbin = np.zeros([Hc,Wc])
    dustbin[np.sum(arr,axis=0) == 0] = 1
    # # 只保留一个4*4的区域内
    # arr[np.sum(arr,axis=0)>1)]
    # assert (np.sum(arr,axis=0)>1).sum() == 0 , print((np.sum(arr,axis=0)>1).sum())
    outarr = np.concatenate([arr,dustbin[np.newaxis,:,:]],axis=0)

    return arr




        
class cls_Dataset(Dataset):
    def __init__(self,txtfile,imagesize = 512, cell_size=4,smooth_factor=1):
        super(cls_Dataset,self).__init__()
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile+'**.tif')
        self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]
        # self.prepare_data() # DeepBlink

    def prepare_data(self) -> None:
        """Convert raw labels into prediction matrices.
        """
        def __convert(dataset, image_size, cell_size):
            labels = []
            for coords in dataset:
                matrix = get_prediction_matrix(coords, image_size, cell_size)
                matrix[..., 0] = np.where( #当matrix[...,0] 中不为0的位置，赋值为smooth_factor,其他位置赋值为1 - self.smooth_factor
                    matrix[..., 0], self.smooth_factor, 1 - self.smooth_factor
                )
                labels.append(matrix)
            return np.array(labels)

        self.label = __convert(self.label, self.image_size, self.cell_size)

        
    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        lb_ = self.label[index]
        lb_img = get_seg_maps(lb_)
        lb_fold = get_fold_label(lb_img,512,4)

        # read
        input_img = cv2.imread(inputpa)

        # norm
        ip_img = func_normlize(input_img[:,:,0],mode='meanstd')
        # lb_img = func_normlize(label_img,mode = 'simple_norm')
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        mask_ = torch.from_numpy(lb_fold).float()
        # return
        name_ = inputpa.split('/')[-1].split('.')[0]
        # ip_lb = (img_,mask_,name_)
        ip_lb = (img_,mask_,name_,input_img)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)
