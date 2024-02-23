# from pyexpat import model
from site import abs_paths
from dataset.dataload import func_getdataloader
from model.choose_net import func_getnetwork
from creterion.iou import func_ioucreterion
from utils.data import get_coordinate_list,get_probabilities,get_coordinates
from creterion.f1 import compute_metrics_once, euclidean_dist
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import time
import scipy
import pandas as pd
from dataset.dataprocess import func_normlize
from config.default import _C as cfg
from config.default import update_config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str,default='./config/inference_demo_coco.yaml')
    # parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    # parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--visthre', type=float, default=0)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


now = int(round(time.time()*1000))
nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))


args = parse_args()
update_config(cfg, args)
# ===================================================================================================
operation = 'inferance' # train trainval inferance
val_datapath = '/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/val_merge/merge_SNR/'
test_datapath = '/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/testdataset/test_MICROTUBULE/SNR2/'
load_epoch = 22
bs = 1
model_mode = 'PointDet'
gpu_list = [3]
ckp_name = '20220706_16_52_01'
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
dataloader_ins_val = func_getdataloader(val_datapath,batch_size=bs,shuffle=False,num_workers=16,training=False)
dataloader_ins_test = func_getdataloader(test_datapath,batch_size=bs,shuffle=False,num_workers=16,training=False)
model_ins = func_getnetwork(model_mode,cfg)

# if use GPU
device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
model_ins.to(device) # 移动模型到cuda
if torch.cuda.device_count() > 1 and len(gpu_list) > 1:
    model_ins = nn.DataParallel(model_ins, device_ids=gpu_list) # 包装为并行风格模型

# load checkpoint
c_checkpoint = torch.load(ckp_path,map_location='cuda:{}'.format(gpu_list[0]))
model_ins.load_state_dict(c_checkpoint["model_state_dict"])
print("==> Loaded pretrianed model checkpoint '{}'.".format(ckp_path))

thre =None
# start inferance
print('====>>>Choose Threshold')
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
    for data in dataloader_ins_val:
        inp = data[0].to(device)
        lab_coords = data[1][0].numpy()
        name = data[2][0]
        inputimage = data[3][0].numpy()


        pheatmap,poffset,psegment = model_ins(inp)
        psegment = psegment[0,0,:,:].detach().cpu().numpy()

        pred_coords = get_coordinates(psegment,thre*0.1)
        # inp = data[0].to(device)
        # lab = data[1][0]
        # lab_coords = get_coordinate_list(lab, image_size=max(inp.shape), probability=0.5)
        # pred = model_ins(inp)[0].permute(1,2,0).detach().cpu().numpy()
        # pred_coords = get_coordinate_list(pred, image_size=max(inp.shape), probability=thre*0.1)

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
    message = 'threthold:{:.1f} f1:{:.3f} precision:{:.3f} recall:{:.3f} rmse:{:.3f} elapse:{:.0f}m {:.0f}s'.format(
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

print('===>Start prediction')

# make save folder
inf_dir = './Log/'+nowname+'/prediction_'+str(thre_max)
if not os.path.exists(inf_dir):
    os.makedirs(inf_dir)

loss_ = 0
f1_list = []
precis_list = []
recall_list = []
abs_euclideans_list = []
for data in dataloader_ins_test:
    inp = data[0].to(device)
    lab_coords = data[1][0].numpy()
    name = data[2][0]
    inputimage = data[3][0].numpy()

    pheatmap,poffset,psegment = model_ins(inp)
    psegment = psegment[0,0,:,:].detach().cpu().numpy()

    pred_coords = get_coordinates(psegment,thre_max)
    # inp = data[0].to(device)
    # lab = data[1][0]
    # name = data[2][0]
    # inputimage = data[3][0].numpy()

    # lab_coords = get_coordinate_list(lab, image_size=max(inp.shape), probability=0.5)
    # pred = model_ins(inp)[0].permute(1,2,0).detach().cpu().numpy()
    # pred_coords = get_coordinate_list(pred, image_size=max(inp.shape), probability=thre_max)
    
    f1_,precis_,recall_,abs_euclideans = compute_metrics_once(pred=pred_coords,true=lab_coords,mdist=3.0)            
    f1_list.append(f1_)
    precis_list.append(precis_)
    recall_list.append(recall_)
    abs_euclideans_list.append(abs_euclideans)

    pred_coords_pd = pd.DataFrame(pred_coords,columns=['pos_y','pos_x'])
    pred_coords_pd.to_csv(inf_dir+'/'+name+'.csv',index = None)

    inputimage = func_normlize(inputimage,mode='maxmin_norm')
    inputimage = np.clip(np.round(inputimage*255),0,255).astype(np.uint8)

    # inputimage[:,:,0] = 0
    # inputimage[:,:,2] = 0

    # cv2.imwrite(inf_dir+'/'+name+'.png',inputimage)
    for y,x in pred_coords:
        cv2.circle(inputimage, (int(y), int(x)), 5, (0, 255, 255), 1)

    for y,x in lab_coords:
        cv2.circle(inputimage, (int(y), int(x)), 1, (0, 0, 255), 1)

    # print('save:'+name)
    cv2.imwrite(inf_dir+'/'+name+'_f1{:.3f}.png'.format(f1_),inputimage)

message = '==>>[TEST]threthold:{:.1f} f1:{:.3f} precision:{:.3f} recall:{:.3f} rmse:{:.3f}'.format(
        thre_max,
        np.array(f1_list).mean(), 
        np.array(precis_list).mean(),
        np.array(recall_list).mean(),
        np.array(abs_euclideans_list).mean()
        )
print(message)
print('=====>>>>'+nowname)
print(test_datapath.split('/')[-3])
print(test_datapath.split('/')[-2])
print(test_datapath.split('/')[-1])