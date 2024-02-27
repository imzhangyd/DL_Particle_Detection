from numpy import core
from torch.utils.data import Dataset
import cv2
import torch
from dataset.dataprocess import func_normlize
import glob
import numpy as np
import pandas as pd
from dataset.enhanceimage import auto_adjust


__author__ = "Yudong Zhang"


def get_seg_maps(arr: np.ndarray, size: int = 512):
    """Convert coordinate list into single-pixel segmentation maps."""

    # 1pixel
    seg_maps = np.zeros((1, size, size))
    for coord in np.round(arr).astype(int):
        seg_maps[0,min(coord[1],size -1 ), min (coord[0],size-1)] = 1
    return seg_maps

    # # 4pixel
    # seg_maps = np.zeros((1, size, size))
    # for coord in np.trunc(arr).astype(int):
    #     seg_maps[0,min(coord[1],size -1 ), min (coord[0],size-1)] = 1
    #     seg_maps[0,min(coord[1]+1,size -1), min(coord[0]+1,size-1)] = 1
    #     seg_maps[0,min(coord[1]+1,size -1), min(coord[0],size-1)] = 1
    #     seg_maps[0,min(coord[1],size -1), min(coord[0]+1,size-1)] = 1
    # return seg_maps


    ## 9 pixel
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
    def __init__(self,txtfile,imagesize = 512, cell_size=4, smooth_factor=1, training= True):
        super(cls_Dataset,self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile+'**.tif')
        self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]

        
    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        lb_ = self.label[index]
        lb_img = get_seg_maps(lb_)
        # read
        input_img = cv2.imread(inputpa)

        # norm
        ip_img = func_normlize(input_img[:,:,0],mode='meanstd')
        # lb_img = func_normlize(label_img,mode = 'simple_norm')
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        mask_ = torch.from_numpy(lb_img).float()
        # return
        name_ = inputpa.split('/')[-1].split('.')[0]
        # ip_lb = (img_,mask_,name_)
        if self.training:
            ip_lb = (img_,mask_,name_,input_img)
        else:
            ip_lb = (img_,lb_,name_,input_img)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)

class cls_Dataset_16(Dataset):
    def __init__(self,txtfile,imagesize = 512, cell_size=4, smooth_factor=1, training= True):
        super(cls_Dataset,self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile+'**.tif')
        self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]

        
    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        lb_ = self.label[index]
        lb_img = get_seg_maps(lb_)
        # read
        input_img = cv2.imread(inputpa,-1)

        # norm
        ip_img = func_normlize(input_img, mode='meanstd')
        # lb_img = func_normlize(label_img,mode = 'simple_norm')
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        mask_ = torch.from_numpy(lb_img).float()
        # return
        name_ = inputpa.split('/')[-1].split('.')[0]
        # ip_lb = (img_,mask_,name_)
        if self.training:
            ip_lb = (img_,mask_,name_,input_img)
        else:
            ip_lb = (img_,lb_,name_,input_img)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)
    
class cls_Dataset_onlypred(Dataset):
    def __init__(self,txtfile,imagesize = 512, cell_size=4, smooth_factor=1, training= True):
        super(cls_Dataset_onlypred,self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile+'**.tif')
        # self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        # self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]

        
    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        # lb_ = self.label[index]
        # lb_img = get_seg_maps(lb_)
        # read
        input_img = cv2.imread(inputpa)
        inputimage = func_normlize(input_img,mode='maxmin_norm')
        inputimage = np.clip(np.round(inputimage*255),0,255).astype(np.uint8)
        # norm
        ip_img = func_normlize(input_img[:,:,0],mode='meanstd')
        # lb_img = func_normlize(label_img,mode = 'simple_norm')
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        # mask_ = torch.from_numpy(lb_img).float()
        # return
        name_ = inputpa.split('/')[-1].split('.')[0]
        # ip_lb = (img_,mask_,name_)
        ip_lb = (img_,name_,inputimage)
        # if self.training:
        #     ip_lb = (img_,mask_,name_,input_img)
        # else:
        #     ip_lb = (img_,lb_,name_,input_img)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)

class cls_Dataset_onlypred_16(Dataset):
    def __init__(self,txtfile,imagesize = 512, cell_size=4, smooth_factor=1, training= True):
        super(cls_Dataset_onlypred_16,self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile+'**.tif')
        # self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        # self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]

        
    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        # lb_ = self.label[index]
        # lb_img = get_seg_maps(lb_)
        # read for vis
        input_img = cv2.imread(inputpa, -1)
        minvalue,maxvalue = auto_adjust(input_img, level = 4) # level 对应level*10像素的bar
        clipimage = np.clip(input_img,minvalue,maxvalue)
        enhanceimg = (clipimage - minvalue) / (maxvalue - minvalue)
        inputimage = (enhanceimg * 255.).astype(np.uint8)
        if len(inputimage.shape) == 2:
            inputimage  = np.repeat(inputimage[:, :, np.newaxis], 3, axis=2)
        # norm for prediction
        ip_img = func_normlize(input_img,mode='meanstd')
        # lb_img = func_normlize(label_img,mode = 'simple_norm')
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        # mask_ = torch.from_numpy(lb_img).float()
        # return
        name_ = inputpa.split('/')[-1].split('.')[0]
        # ip_lb = (img_,mask_,name_)
        ip_lb = (img_,name_,inputimage)
        # if self.training:
        #     ip_lb = (img_,mask_,name_,input_img)
        # else:
        #     ip_lb = (img_,lb_,name_,input_img)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)