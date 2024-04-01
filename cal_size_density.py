''''
Use Gaussian fitting to correct the position,
and calculate the intensity and diameter of particles.
Output new Dataframe

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
    
    parser.add_argument('--det_folder', type=str, default='./Log/20240327_09_54_10_lamp_vesicle4_eval/prediction_0.99')
    
    # param
    # parser.add_argument('--diameter_filter', type=float, default=0.0)
    # parser.add_argument('--density_filter', type=float, default=0.0)
    parser.add_argument('--gaussfit_search_radius', type=int, default=4)

    parser.add_argument('--img_folder', type=str, default='/mnt/data1/ZYDdata/lwj/200609 lamp-1mch int2s 013 ori')
    parser.add_argument('--save_folder',type=str, default='./Log/20240327_09_54_10_lamp_vesicle4_eval/prediction_0.99_cal1')
    # parser.add_argument('--gaussfit_diameter')
    args = parser.parse_args()

    return args


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


if __name__ == "__main__":
    opt = parse_args_()
    r_S = opt.gaussfit_search_radius
    savefo = opt.save_folder
    os.makedirs(savefo, exist_ok=True)
    # diameter_thre = opt.diameter_filter
    # density_thre = opt.density_filter
    
    # save path
    # savepath = opt.det_folder+'_filter_di{:.1f}_de{:.1f}'.format(opt.diameter_filter, opt.density_filter)

    allcsvresultpathlist = glob.glob(os.path.join(opt.det_folder, '**.csv'))
    allcsvresultpathlist.sort()
    # alldiameter = []
    # alldensity = []
    for csvpa in allcsvresultpathlist:
        # add new col 

        df = pd.read_csv(csvpa, header=0, index_col=None)
        df['diameter'] = 0.0
        df['density'] = 0.0

        # read the frame
        csvname = os.path.split(csvpa)[-1]
        imgname = csvname.replace('.csv','.tif')
        imgpa = os.path.join(opt.img_folder,imgname)
        # imgpa = csvpa[:-3]+'png'

        theimg = cv2.imread(imgpa)
        theimg = cv2.cvtColor(theimg, cv2.COLOR_BGR2GRAY)
        theimg = cv2.cvtColor(theimg, cv2.COLOR_GRAY2BGR)
        
        theimg = theimg[:,:,0] #yellow circle in the last two channels
        imgheight, imgwidth = theimg.shape
        # print(imgheight,imgwidth)
        # iterate each point 
        for i in df.index:
            left = df.iloc[i].loc['pos_x']
            top = df.iloc[i].loc['pos_y']
            # print(left,top,r_S)
            l = max(int(left)-r_S, 0)
            r = min(int(left)+r_S+1, imgwidth - 1)
            t = max(int(top)-r_S, 0)
            b = min(int(top)+r_S+1, imgheight - 1)
            # print(int(top)+r_S+1)
            # print(t,b,l,r)
            # Create the gaussian data
            # Xin, Yin = np.mgrid[0:201, 0:201]
            # data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + np.random.random(Xin.shape)
            data = theimg[t:b, l:r]
            params = fitgaussian(data)
            (height, x, y, width_x, width_y) = params # x==top' y==left'
            x = t+x
            y = l+y

            # 判断是否有gauss点，如果拟合的坐标在外面，先绘制一下看看有哪些情况
            if x<t or y<l or x>b or y>r or width_x is None or width_y is None or np.isnan(width_x) or np.isnan(width_y):
                
                if np.isnan(width_x) or np.isnan(width_y): print(f'{i} width is nan')
                # plt.matshow(data)#, cmap=plt.cm.gist_earth_r)
                # fit = gaussian(*params)

                # plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
                # ax = plt.gca()
            
                # plt.text(0.95, 0.05, """
                # x : %.1f
                # y : %.1f
                # orix: %.1f
                # oriy: %.1f
                # width_x : %.1f
                # width_y : %.1f""" %(x, y, top,left,width_x, width_y),
                #         fontsize=16, horizontalalignment='right',
                #         verticalalignment='bottom', transform=ax.transAxes)
                # plt.savefig(os.path.join(savefo,f'{csvname[:-4]}-{i}.png'))
                # plt.close()
                df.at[i, 'diameter'] = -1.0
                df.at[i, 'density'] = theimg[int(top),int(left)]*2
            else:
                # save new csv
                df.at[i, 'pos_x'] = y
                df.at[i, 'pos_y'] = x
                df.at[i, 'diameter'] = min(width_x, width_y)*2
                df.at[i, 'density'] = theimg[int(x),int(y)]*2


        # save filter results

        df.to_csv(os.path.join(savefo, csvname), index=False)



        
