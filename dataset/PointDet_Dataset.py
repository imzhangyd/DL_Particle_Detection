from torch.utils.data import Dataset
import cv2
import torch
from dataset.dataprocess import func_normlize
import glob
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from utils.data import get_prediction_matrix
# from dataset.target_generators import HeatmapGenerator, OffsetGenerator


__author__ = "Yudong Zhang"


def get_seg_maps(arr: np.ndarray, size: int = 512):
    """Convert coordinate list into single-pixel segmentation maps."""

    # 1pixel
    seg_maps = np.zeros((1, size, size))
    for coord in np.round(arr).astype(int):
        seg_maps[0,min(coord[1],size -1 ), min (coord[0],size-1)] = 1
    return seg_maps

    # 4pixel
    # seg_maps = np.zeros((1, size, size))
    # for coord in np.trunc(arr).astype(int):
    #     seg_maps[0,min(coord[1],size -1 ), min (coord[0],size-1)] = 1
    #     seg_maps[0,min(coord[1]+1,size -1), min(coord[0]+1,size-1)] = 1
    #     seg_maps[0,min(coord[1]+1,size -1), min(coord[0],size-1)] = 1
    #     seg_maps[0,min(coord[1],size -1), min(coord[0]+1,size-1)] = 1
    # return seg_maps


    # ## 9 pixel
    # seg_maps = np.zeros((1,size,size))
    # for coord in np.trunc(arr).astype(int):

    #     seg_maps[0,coord[1], coord[0]] = 1
    #     seg_maps[0,min(coord[1]+1,size -1), min(coord[0]+1,size-1)] = 1
    #     seg_maps[0,min(coord[1]+1,size -1), coord[0]] = 1
    #     seg_maps[0,coord[1], min(coord[0]+1,size-1)] = 1

    #     seg_maps[0,max(coord[1]-1,0), max(coord[0]-1,0)] = 1
    #     seg_maps[0,max(coord[1]-1,0), coord[0]] = 1
    #     seg_maps[0,coord[1], max(coord[0]-1,0)] = 1

    #     seg_maps[0,max(coord[1]-1,0), min(coord[0]+1,size-1)] = 1
    #     seg_maps[0,min(coord[1]+1,size-1), max(coord[0]-1,0)] = 1

    # return seg_maps

        
class cls_Dataset(Dataset):
    def __init__(self,txtfile, imagesize = 512,training=True):
        super(cls_Dataset,self).__init__()
        self.image_size = imagesize
        self.imagepathlist = glob.glob(txtfile+'**.tif')
        self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]
        self.sigma = 10.0
        self.center_sigma = 4.0
        self.bg_weight = 0.1
        self.training = training

        
    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        # read
        input_img = cv2.imread(inputpa)
        ip_img = func_normlize(input_img[:,:,0],mode='meanstd')
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()

        lb_ = self.label[index]

        name_ = inputpa.split('/')[-1].split('.')[0]

        if self.training:
            # generate mask4heatmap
            mask_cls = get_seg_maps(lb_)

            # generate heatmap
            # heatmap, ignored = self.heatmap_generator( #ignored 0.1 1
            #     lb_, self.sigma, self.center_sigma, self.bg_weight)

            heatmap = mask_cls
            ignored = mask_cls
            
            # generate offsetmap
            # offset, offset_weight = self.offset_generator(lb_)
            offset = mask_cls
            offset_weight = mask_cls

            # norm
            
            # lb_img = func_normlize(label_img,mode = 'simple_norm')
            # numpy->torch
            
            mask_ = torch.from_numpy(mask_cls).float()


            # return
            
            # ip_lb = (img_,mask_,name_)
            ip_lb = (img_,mask_,heatmap, ignored, offset, offset_weight, name_,input_img)
            return ip_lb
        else:
            return (img_,lb_,name_,input_img)

    def __len__(self):
        return len(self.imagepathlist)


class cls_Dataset_16(Dataset):
    def __init__(self,txtfile, imagesize = 512,training=True):
        super(cls_Dataset_16,self).__init__()
        self.image_size = imagesize
        self.imagepathlist = glob.glob(txtfile+'**.tif')
        self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]
        self.sigma = 10.0
        self.center_sigma = 4.0
        self.bg_weight = 0.1
        self.training = training


    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        # read
        input_img = cv2.imread(inputpa, -1)
        ip_img = func_normlize(input_img,mode='meanstd')
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()

        lb_ = self.label[index]

        name_ = inputpa.split('/')[-1].split('.')[0]

        if self.training:
            # generate mask4heatmap
            mask_cls = get_seg_maps(lb_)

            # generate heatmap
            # heatmap, ignored = self.heatmap_generator( #ignored 0.1 1
            #     lb_, self.sigma, self.center_sigma, self.bg_weight)

            heatmap = mask_cls
            ignored = mask_cls
            
            # generate offsetmap
            # offset, offset_weight = self.offset_generator(lb_)
            offset = mask_cls
            offset_weight = mask_cls

            # norm
            
            # lb_img = func_normlize(label_img,mode = 'simple_norm')
            # numpy->torch
            
            mask_ = torch.from_numpy(mask_cls).float()

            # return
            
            # ip_lb = (img_,mask_,name_)
            ip_lb = (img_,mask_,heatmap, ignored, offset, offset_weight, name_,input_img)
            return ip_lb
        else:
            return (img_,lb_,name_,input_img)

    def __len__(self):
        return len(self.imagepathlist)
    


if __name__ == '__main__':
    pass