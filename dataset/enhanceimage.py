import numpy as np
import cv2
import os
# import matplotlib.pyplot as plt
import glob


__author__ = "Yudong Zhang"


AUTO_THRESHOLD = 1

def auto_adjust(imp, level=2):
    # (histogram,bins,_)= plt.hist(imp.flatten(),bins=256)
    (histogram,bins)= np.histogram(imp.flatten(),bins=256)
    H,W = imp.shape
    imp_min = imp.flatten().min()
    imp_max = imp.flatten().max()
    pixels_number = H*W
    limit = pixels_number // 10
    autoThreshold = pixels_number // 10
    if autoThreshold < 10:
        autoThreshold = AUTO_THRESHOLD
    else:
        autoThreshold /= level

    threshold = pixels_number // autoThreshold

    i = -1
    found = False
    count = 0
    while not found and i < 255:
        i += 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
       
    hmin = i
    
    i = 256
    found = False
    count = 0
    while not found and i > 0:
        i -= 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
       
    hmax = i
    if hmax > hmin:
        
        min_value = bins[hmin]
        max_value = bins[hmax]
        
    else:
        min_value = imp_min
        max_value = imp_max

    return min_value,max_value


if __name__ =='__main__':
    imgfolder = 'C:/Users/pc/Downloads/doi_10.5061_dryad.0zpc866wh__v4/CDX2_movies_and_evaluation/CDX2'
    imgpathlist = glob.glob(os.path.join(imgfolder,'**.tif'))
    savefolder = 'C:/Users/pc/Downloads/doi_10.5061_dryad.0zpc866wh__v4/CDX2_movies_and_evaluation/CDX2_vis'
    os.makedirs(savefolder, exist_ok=True)
    for imgpath in imgpathlist:
        imgname = os.path.split(imgpath)[-1]
        image = cv2.imread(imgpath,-1)
        minvalue,maxvalue = auto_adjust(image, level = 4)
        clipimage = np.clip(image,minvalue,maxvalue)
        enhanceimg = (clipimage - minvalue) / (maxvalue - minvalue)
        enhanceimg = (enhanceimg * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(savefolder, imgname),enhanceimg)