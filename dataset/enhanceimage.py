import numpy as np
import cv2
import os
# import matplotlib.pyplot as plt
import glob


AUTO_THRESHOLD = 1

def auto_adjust(imp, level=2):
    # (histogram,bins,_)= plt.hist(imp.flatten(),bins=256)
    (histogram,bins)= np.histogram(imp.flatten(),bins=256)
    H,W = imp.shape
    imp_min = imp.flatten().min()
    imp_max = imp.flatten().max()
    pixels_number = H*W
    limit = pixels_number // 10
    autoThreshold = pixels_number // 10  # 初始化阈值参数
    if autoThreshold < 10:
        autoThreshold = AUTO_THRESHOLD  # 设置默认阈值参数
    else:
        autoThreshold /= level  # 阈值参数减半

    threshold = pixels_number // autoThreshold  # 根据阈值参数计算阈值

    i = -1
    found = False
    count = 0
    while not found and i < 255:
        i += 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold  # 从左到右找到满足阈值条件的最小直方图值
       
    hmin = i  # 最小直方图值对应的像素值
    
    i = 256
    found = False
    count = 0
    while not found and i > 0:
        i -= 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold  # 从右到左找到满足阈值条件的最大直方图值
       
    hmax = i  # 最大直方图值对应的像素值
    if hmax > hmin:  # 如果找到了有效的最小和最大直方图值对应的像素值
        
        min_value = bins[hmin]  # 计算最小像素值
        max_value = bins[hmax]  # 计算最大像素值
        
    else:
        min_value = imp_min
        max_value = imp_max  # 如果最小和最大像素值相等，则使用统计信息中的最小和最大像素值

    return min_value,max_value


if __name__ =='__main__':
    imgfolder = 'C:/Users/pc/Downloads/doi_10.5061_dryad.0zpc866wh__v4/CDX2_movies_and_evaluation/CDX2'
    imgpathlist = glob.glob(os.path.join(imgfolder,'**.tif'))
    savefolder = 'C:/Users/pc/Downloads/doi_10.5061_dryad.0zpc866wh__v4/CDX2_movies_and_evaluation/CDX2_vis'
    os.makedirs(savefolder, exist_ok=True)
    for imgpath in imgpathlist:
        imgname = os.path.split(imgpath)[-1]
        image = cv2.imread(imgpath,-1)
        minvalue,maxvalue = auto_adjust(image, level = 4) # level 对应level*10像素的bar
        clipimage = np.clip(image,minvalue,maxvalue)
        enhanceimg = (clipimage - minvalue) / (maxvalue - minvalue)
        enhanceimg = (enhanceimg * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(savefolder, imgname),enhanceimg)