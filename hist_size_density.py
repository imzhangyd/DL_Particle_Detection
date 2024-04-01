''''
Output hist of diameter and intensity for filter next.

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
from scipy import optimize



def parse_args_():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--postdet_folder', type=str, default='./Log/20240327_09_54_10_lamp_vesicle4_eval/prediction_0.99_cal1')
    
    parser.add_argument('--save_folder',type=str, default='./Log/20240327_09_54_10_lamp_vesicle4_eval/prediction_0.99_cal1')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    opt = parse_args_()
    savefo = opt.save_folder
    os.makedirs(savefo, exist_ok=True)

    allcsvresultpathlist = glob.glob(os.path.join(opt.postdet_folder, '**.csv'))
    allcsvresultpathlist.sort()
    alldiameter = []
    alldensity = []
    for csvpa in allcsvresultpathlist:
        # add new col 
        print(csvpa)
        df = pd.read_csv(csvpa, header=0)
    
        alldiameter+=df['diameter'].values.tolist()
        alldensity+=df['density'].values.tolist()
    
    alldiameter_np = np.array(alldiameter).flatten()
    print('diameter max and min:')
    print(alldiameter_np.max(),alldiameter_np.min())
    plt.figure()
    plt.hist(alldiameter_np,100,range=(0,20))
    plt.savefig(os.path.join(savefo,'diameter_hist_.png'))
    plt.close()
    alldensity_np = np.array(alldensity).flatten()
    print('density max and min:')
    print(alldensity_np.max(),alldensity_np.min())
    plt.figure()
    plt.hist(alldensity_np, 100)
    plt.yscale('log')
    plt.savefig(os.path.join(savefo, 'density_hist_.png'))
    plt.close()
