# from pyexpat import model
from dataset.dataload import func_getdataloader
from model.choose_net import func_getnetwork
from creterion.iou import func_ioucreterion
from utils.data import get_coordinates,get_probabilities
from creterion.f1 import compute_metrics_once
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import time
import scipy
import pandas as pd
from dataset.dataprocess import func_normlize
now = int(round(time.time()*1000))
nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))

# ===================================================================================================
operation = 'inferance' # train trainval inferance
test_datapath = '/ldap_shared/shared/zyd/data/20220611_detparticle/testdataset/test_MICROTUBULE/SNR1/'
load_epoch = 52
bs = 1
model_mode = 'DetNet'
gpu_list = [0]
ckp_name = '20220614_16_33_16'
ckp_path = os.path.join('./Log/',ckp_name+ "/checkpoints/checkpoints_" + str(load_epoch) + ".pth")
# ====================================================================================================

# record log
logtxt_path = './Log/log.txt'
logtxt = open(logtxt_path,'a+')
logtxt.write('\n\n')
logtxt.write('=============={}===============\n'.format(nowname))
logtxt.write('operation={}\n'.format(operation))
logtxt.write('test_datapath={}\n'.format(test_datapath))
logtxt.write('ckp_name={}\n'.format(ckp_name))
logtxt.write('load_epoch={}\n'.format(load_epoch))
logtxt.write('batchsize={}\n'.format(bs))
logtxt.write('model_mode={}\n'.format(model_mode))
logtxt.write('--------------------------------\n')
logtxt.close()


# load data model

dataloader_ins = func_getdataloader(test_datapath,batch_size=bs,shuffle=False,num_workers=16  )
model_ins = func_getnetwork(model_mode)

# if use GPU
device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
model_ins.to(device) # 移动模型到cuda
if torch.cuda.device_count() > 1 and len(gpu_list) > 1:
    model_ins = nn.DataParallel(model_ins, device_ids=gpu_list) # 包装为并行风格模型

# load checkpoint
c_checkpoint = torch.load(ckp_path)
model_ins.load_state_dict(c_checkpoint["model_state_dict"])
print("==> Loaded pretrianed model checkpoint '{}'.".format(ckp_path))

thre =None
# start inferance
model_ins.eval()
f1_max = 0
thre_max = 0.1
for thre in range(1,10):
    loss_ = 0
    f1_list = []
    precis_list = []
    recall_list = []
    abs_euclideans_list = []

    since = time.time()
    for data in dataloader_ins:
        inp = data[0].to(device)
        lab = data[1][0]
        lab_coords = get_coordinates(lab, thre=0.5)
        pred = model_ins(inp)[0].permute(1,2,0).detach().cpu().numpy()
        pred_coords = get_coordinates(pred, thre=thre*0.1)

        f1_,precis_,recall_,abs_euclideans = compute_metrics_once(pred=pred_coords,true=lab_coords,mdist=3.0)            
        f1_list.append(f1_)
        precis_list.append(precis_)
        recall_list.append(recall_)
        abs_euclideans_list.append(abs_euclideans)

        # for _ in iou:
        #     epochiou_list.append(_.cpu().numpy())

    if np.array(f1_list).mean() > f1_max:
        f1_max = np.array(f1_list).mean()
        thre_max = thre*0.1

    # claculate time
    time_elapsed = time.time() - since
    # record loss time 
    message = 'threthold:{:.1f} f1:{:.3f} precision{:.3f} recall{:.3f} rmse{:.3f} elapse:{:.0f}m {:.0f}s'.format(
        thre*0.1,np.array(f1_list).mean(), 
        np.array(precis_list).mean(),
        np.array(recall_list).mean(),
        np.array(abs_euclideans_list).mean(),
        time_elapsed // 60, 
        time_elapsed % 60)
    print(message)
    logtxt = open(logtxt_path,'a+')
    logtxt.write(message+'\n')
    logtxt.close()

print('best f1:{:.3f},with threshold:{:.1f}'.format(f1_max,thre_max))


# make save folder
inf_dir = './Log/'+nowname+'/prediction_'+str(thre_max)
if not os.path.exists(inf_dir):
    os.makedirs(inf_dir)

loss_ = 0

for data in dataloader_ins:
    inp = data[0].to(device)
    lab = data[1][0]
    name = data[2][0]
    inputimage = data[3][0].numpy()

    lab_coords = get_coordinates(lab,thre=0.5)
    pred = model_ins(inp)[0].permute(1,2,0).detach().cpu().numpy()
    pred_coords = get_coordinates(pred,thre=thre_max)

    f1_,precis_,recall_,abs_euclideans = compute_metrics_once(pred=pred_coords,true=lab_coords,mdist=3.0)            

    inputimage = func_normlize(inputimage,mode='maxmin_norm')
    inputimage = np.clip(np.round(inputimage*255),0,255).astype(np.uint8)

    # inputimage[:,:,0] = 0
    # inputimage[:,:,2] = 0

    # cv2.imwrite(inf_dir+'/'+name+'.png',inputimage)
    for x,y in pred_coords:
        cv2.circle(inputimage, (int(y), int(x)), 5, (0, 255, 255), 1)

    for x,y in lab_coords:
        cv2.circle(inputimage, (int(y), int(x)), 1, (0, 0, 255), 1)


    # print('save:'+name)
    cv2.imwrite(inf_dir+'/'+name+'_f1{:.3f}.png'.format(f1_),inputimage)

    
    # for nu,thisf1 in enumerate(f1_list):
    #     print('save:'+name[nu])
    #     cv2.imwrite(inf_dir+'/'+name[nu]+'_iou{:.3f}.png'.format(thisf1),outputimg[nu,0,:,:])
    # break
print('best f1:{:.3f},with threshold:{:.1f}'.format(f1_max,thre_max))


