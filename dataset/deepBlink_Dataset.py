from torch.utils.data import Dataset
import cv2
import torch
from dataset.dataprocess import func_normlize
import glob
import numpy as np
import pandas as pd

from utils.data import get_prediction_matrix
from dataset.enhanceimage import auto_adjust


__author__ = "Yudong Zhang"


class cls_Dataset_16(Dataset):
    def __init__(
        self, txtfile, imagesize=512, cell_size=4, smooth_factor=1, training=True
    ):
        super(cls_Dataset_16, self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile + "**.tif")
        self.labelpathlist = [_.replace("tif", "csv") for _ in self.imagepathlist]
        self.label = [
            pd.read_csv(labelpa, header=0).values[:, :2].astype(float)
            for labelpa in self.labelpathlist
        ]
        self.prepare_data()

    def prepare_data(self) -> None:
        """Convert raw labels into prediction matrices."""

        def __convert(dataset, image_size, cell_size):
            labels = []
            for coords in dataset:
                matrix = get_prediction_matrix(coords, image_size, cell_size)
                matrix[..., 0] = np.where(
                    matrix[..., 0], self.smooth_factor, 1 - self.smooth_factor
                )
                labels.append(matrix)
            return np.array(labels)

        self.labelimg = __convert(self.label, self.image_size, self.cell_size)

    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        lb_img = self.labelimg[index]
        lb_ = self.label[index]
        # read
        input_img = cv2.imread(inputpa, -1)
        # inputimage = func_normlize(input_img,mode='maxmin_norm')
        # inputimage = np.clip(np.round(inputimage*255),0,255).astype(np.uint8)
        # norm
        ip_img = func_normlize(input_img, mode="meanstd")
        # lb_img = func_normlize(label_img,mode = 'simple_norm')
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        mask_ = torch.from_numpy(lb_img).float()
        # return
        name_ = inputpa.split("/")[-1].split(".")[0]
        # ip_lb = (img_,mask_,name_)
        if self.training:
            ip_lb = (img_, mask_)
        else:
            minvalue, maxvalue = auto_adjust(
                input_img, level=4
            )  # level 对应level*10像素的bar
            clipimage = np.clip(input_img, minvalue, maxvalue)
            enhanceimg = (clipimage - minvalue) / (maxvalue - minvalue)
            inputimage = (enhanceimg * 255.0).astype(np.uint8)
            ip_lb = (img_, mask_, name_, inputimage)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)


class cls_Dataset(Dataset):
    def __init__(
        self, txtfile, imagesize=512, cell_size=4, smooth_factor=1, training=True
    ):
        super(cls_Dataset, self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile + "**.tif")
        self.labelpathlist = [_.replace("tif", "csv") for _ in self.imagepathlist]
        self.label = [
            pd.read_csv(labelpa, header=None).values[:, :2]
            for labelpa in self.labelpathlist
        ]
        self.prepare_data()

    def prepare_data(self) -> None:
        """Convert raw labels into prediction matrices."""

        def __convert(dataset, image_size, cell_size):
            labels = []
            for coords in dataset:
                matrix = get_prediction_matrix(coords, image_size, cell_size)
                matrix[..., 0] = np.where(
                    matrix[..., 0], self.smooth_factor, 1 - self.smooth_factor
                )
                labels.append(matrix)
            return np.array(labels)

        self.labelimg = __convert(self.label, self.image_size, self.cell_size)

    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        lb_img = self.labelimg[index]
        lb_ = self.label[index]
        # read
        input_img = cv2.imread(inputpa)

        # norm
        ip_img = func_normlize(input_img[:, :, 0], mode="meanstd")
        # lb_img = func_normlize(label_img,mode = 'simple_norm')
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        mask_ = torch.from_numpy(lb_img).float()
        # return
        name_ = inputpa.split("/")[-1].split(".")[0]
        # ip_lb = (img_,mask_,name_)
        if self.training:
            ip_lb = (img_, mask_, name_, input_img)
        else:
            ip_lb = (img_, lb_, name_, input_img)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)


class cls_Dataset_onlypred(Dataset):
    def __init__(
        self, txtfile, imagesize=512, cell_size=4, smooth_factor=1, training=True
    ):
        super(cls_Dataset_onlypred, self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile + "**.tif")
        # self.labelpathlist = [_.replace('tif','csv') for _ in self.imagepathlist]
        # self.label = [pd.read_csv(labelpa,header=None).values[:,:2] for labelpa in self.labelpathlist]
        # self.prepare_data()

    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        # lb_img = self.labelimg[index]
        # lb_ = self.label[index]
        # read
        input_img = cv2.imread(inputpa)
        inputimage = func_normlize(input_img, mode="maxmin_norm")
        inputimage = np.clip(np.round(inputimage * 255), 0, 255).astype(np.uint8)

        inputimage = cv2.cvtColor(inputimage, cv2.COLOR_BGR2GRAY)
        inputimage = cv2.cvtColor(inputimage, cv2.COLOR_GRAY2BGR)
        assert len(input_img.shape) == 3
        # norm
        if input_img[:, :, 2].max() == 255:
            ip_img = func_normlize(input_img[:, :, 2], mode="meanstd")
        elif input_img[:, :, 1].max() == 255:
            ip_img = func_normlize(input_img[:, :, 1], mode="meanstd")
        elif input_img[:, :, 0].max() == 255:
            ip_img = func_normlize(input_img[:, :, 0], mode="meanstd")
        else:
            raise ValueError
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        # mask_ = torch.from_numpy(lb_img).float()
        # return
        name_ = inputpa.split("/")[-1].split(".")[0]
        # ip_lb = (img_,mask_,name_)
        ip_lb = (img_, name_, inputimage)
        # if self.training:
        #     ip_lb = (img_,mask_,name_,input_img)
        # else:
        #     ip_lb = (img_,lb_,name_,input_img)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)


class cls_Dataset_onlypred_16(Dataset):
    def __init__(
        self, txtfile, imagesize=512, cell_size=4, smooth_factor=1, training=True
    ):
        super(cls_Dataset_onlypred_16, self).__init__()
        self.training = training
        self.image_size = imagesize
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor
        self.imagepathlist = glob.glob(txtfile + "**.tif")

    def __getitem__(self, index: int):
        # get path
        inputpa = self.imagepathlist[index]
        # read
        input_img = cv2.imread(inputpa, -1)
        if len(input_img.shape) == 3:
            input_img = input_img[:, :, 0]
        # ipdb.set_trace()
        minvalue, maxvalue = auto_adjust(
            input_img, level=4
        )  # level 对应level*10像素的bar
        clipimage = np.clip(input_img, minvalue, maxvalue)
        enhanceimg = (clipimage - minvalue) / (maxvalue - minvalue)
        inputimage = (enhanceimg * 255.0).astype(np.uint8)
        if len(inputimage.shape) == 2:
            inputimage = np.repeat(inputimage[:, :, np.newaxis], 3, axis=2)

        # norm
        ip_img = func_normlize(input_img, mode="meanstd")
        # numpy->torch
        img_ = torch.from_numpy(ip_img).unsqueeze(dim=0).float()
        # return
        name_ = inputpa.split("/")[-1].split(".")[0]
        # ip_lb = (img_,mask_,name_)
        ip_lb = (img_, name_, inputimage)
        return ip_lb

    def __len__(self):
        return len(self.imagepathlist)
