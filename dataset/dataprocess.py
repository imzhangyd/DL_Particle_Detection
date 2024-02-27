# from torchvision import transforms
import cv2
import numpy as np


__author__ = "Yudong Zhang"


def func_normlize(image,mode = 'simple_norm'):

    if mode == 'simple_norm':
        image = image/255.0
        return image

    elif mode == 'maxmin_norm':
        image = image-image.flatten().min()
        image = image/image.flatten().max()
        return image
    
    elif mode == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        image = image/255.0
        return image

    elif mode == 'meanstd':
        image = (image-image.mean()) / image.std()
        return image.astype(np.float32)