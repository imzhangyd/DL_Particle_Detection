import glob
import pandas as pd
import numpy as np
import argparse
import os


__author__ = "Yudong Zhang"


def readXML(file):
    with open(file) as f:
        lines = f.readlines()
    f.close()
    poslist = []
    p = 0
    for i in range(len(lines)):
        if '<particle>' in lines[i]:
            posi = []
        elif '<detection t=' in lines[i]:
            ind1 = lines[i].find('"')
            ind2 = lines[i].find('"', ind1 + 1)
            t = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            x = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            y = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            z = float(lines[i][ind1 + 1:ind2])
            posi.append([x, y, t, z, float(p)])
        elif '</particle>' in lines[i]:
            p += 1
            poslist.append(posi)
    return poslist



def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--imgfolder', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low')
    parser.add_argument('--track_xmlpath', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low/MICROTUBULE snr 7 density low.xml')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()

    imagelist = glob.glob(os.path.join(opt.imgfolder, '**.tif'))
    xmlpa = opt.track_xmlpath
    print('====START=====>>'+str(len(imagelist)))

    poslist = readXML(xmlpa)

    for pa in imagelist:
        frame = int(pa.split('/')[-1].split(' t')[1].split(' ')[0])
        P = [np.array(_) for _ in poslist]
        M = np.vstack(P)
        detection_total = pd.DataFrame(M[:,:3])
        detection_total.columns=['pos_x','pos_y','frame']
        pos_csv = detection_total[detection_total['frame'] == frame]
        pos_csv.to_csv(pa.replace('tif','csv'),index = None, header = None)
    print('====FINISH=====>>'+str(len(imagelist)))
