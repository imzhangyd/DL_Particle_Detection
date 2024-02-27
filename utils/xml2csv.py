import glob
import pandas as pd
import numpy as np

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


# datatype = 'train'
# imagelist = glob.glob('./Data/'+datatype+'/**.tif')
xmlroot = "/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/xml/"

for SNR_iterm in [1,2,4,7]:
    imagelist = glob.glob(
        "/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/val_VESICLE/SNR"+str(SNR_iterm)+"/**.tif")
    print('====START=====>>'+str(len(imagelist)))
    for pa in imagelist:
        print(pa)
        name = pa.split('/')[-1].split(' t')[0]
        xmlpa = xmlroot+name+'.xml'
        frame = int(pa.split('/')[-1].split(' t')[1].split(' ')[0])
        # print(frame)
        poslist = readXML(xmlpa)
        # print(poslist)
        
        P = [np.array(_) for _ in poslist]
        M = np.vstack(P)
        detection_total = pd.DataFrame(M[:,:3])
        detection_total.columns=['pos_x','pos_y','frame']
        # print(detection_total)
        pos_csv = detection_total[detection_total['frame'] == frame]
        # print(pos_csv)
        pos_csv.to_csv(pa.replace('tif','csv'),index = None, header = None)
    print('====FINISH=====>>'+str(len(imagelist)))

# imagelist = glob.glob("/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/val_merge/SNR2/**.tif")
# print('=========>>'+str(len(imagelist)))
# for pa in imagelist:
#     print(pa)
#     name = pa.split('/')[-1].split(' t')[0]
#     xmlpa = xmlroot+name+'.xml'
#     frame = int(pa.split('/')[-1].split(' t')[1].split(' ')[0])
#     # print(frame)
#     poslist = readXML(xmlpa)
#     # print(poslist)
#     # all detection df
#     P = [np.array(_) for _ in poslist]
#     M = np.vstack(P)
#     detection_total = pd.DataFrame(M[:,:3])
#     detection_total.columns=['pos_x','pos_y','frame']
#     # print(detection_total)
#     pos_csv = detection_total[detection_total['frame'] == frame]
#     # print(pos_csv)
#     pos_csv.to_csv(pa.replace('tif','csv'),index = None, header = None)
# print('=========>>'+str(len(imagelist)))