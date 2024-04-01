''''
Filter by diameter and density and visualize.
'''
import numpy as np
import os
import argparse
import glob
import pandas as pd
import cv2
import numpy as np
# from astrops.io import fits as pf
import matplotlib.pyplot as plt
import os
import sys
# from astropy.wcs import WCS
import pandas as pd
# import matplotlib.patches as patches
# from scipy import optimize
from utils.NMS import nms_point



def parse_args_():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--det_folder', type=str, default='./Log/20240327_09_54_10_lamp_vesicle4_eval/prediction_0.99_cal1')
    
    # param
    parser.add_argument('--diameter_filter', type=float, default=2.0)
    parser.add_argument('--density_filter', type=float, default=200.0)

    parser.add_argument('--img_folder', type=str, default='/mnt/data1/ZYDdata/lwj/200609 lamp-1mch int2s 013 ori')
    parser.add_argument('--NMS_dist', type=float, default=5.0)
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    opt = parse_args_()
    diameter_thre = opt.diameter_filter
    density_thre = opt.density_filter
    
    # save path
    savepath = opt.det_folder+'_filter_di{:.1f}_de{:.1f}'.format(opt.diameter_filter, opt.density_filter)
    os.makedirs(savepath,exist_ok=True)

    
    allcsvresultpathlist = glob.glob(os.path.join(opt.det_folder, '**.csv'))
    allcsvresultpathlist.sort()
    for csvpa in allcsvresultpathlist:

        df = pd.read_csv(csvpa, header=0)
        
        csvname = os.path.split(csvpa)[-1]
        name = csvname[:-4]
        imgname = csvname.replace('.csv','.tif')
        imgpa = os.path.join(opt.img_folder,imgname)

        theimg = cv2.imread(imgpa)
        theimg = cv2.cvtColor(theimg, cv2.COLOR_BGR2GRAY)
        theimg = cv2.cvtColor(theimg, cv2.COLOR_GRAY2BGR)


        # filter
        newdf = df[df['diameter'] > diameter_thre]
        newdf = newdf[newdf['density'] > density_thre]

        # NMS
        scores = newdf['density'].values
        # thresh = 25
        newpoints_index = nms_point(newdf[['pos_y','pos_x']].values, scores, opt.NMS_dist**2)
        newdf = newdf.iloc[newpoints_index]

        # save filter results
        newdf.to_csv(os.path.join(savepath, csvname), index=False)

        # vis
        for y,x in newdf[['pos_y','pos_x']].values:
            cv2.circle(theimg, (int(x), int(y)), 5, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(savepath, name+'.png'),theimg)

        # break

        
