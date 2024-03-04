import glob
import pandas as pd
import os
import argparse
import shutil

__author__ = "Yudong Zhang"


def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_folder', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low')
    parser.add_argument('--save_path', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low')
    parser.add_argument('--split_p', type=float, default=0.8) 

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()

    os.makedirs(opt.save_path+'_train', exist_ok=True)
    os.makedirs(opt.save_path+'_val', exist_ok=True)
    totalimgpath = glob.glob(os.path.join(opt.img_folder, '**.tif'))
    totalimgpath.sort()

    trainnum = int(len(totalimgpath)*0.8)

    for i, imgpa in enumerate(totalimgpath):
        imgfilename = os.path.split(imgpa)[-1]
        imglabelname = imgfilename.replace('.tif','.csv')
        if i < trainnum:
            shutil.copy(imgpa, os.path.join(opt.save_path+'_train', imgfilename))
            shutil.copy(imgpa.replace('.tif','.csv'), os.path.join(opt.save_path+'_train', imglabelname))
        else:
            shutil.copy(imgpa, os.path.join(opt.save_path+'_val', imgfilename))
            shutil.copy(imgpa.replace('.tif','.csv'), os.path.join(opt.save_path+'_val', imglabelname))


