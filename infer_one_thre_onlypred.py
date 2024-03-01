"""
Utilize one identified threshold to predict on the test set without evaluation.
"""
from dataset.dataload import func_getdataloader,func_getdataloader_16, func_getdataloader_pred, func_getdataloader_pred_16
from model.choose_net import func_getnetwork
from utils.data import get_coordinate_list
from utils.data import get_coordinates,get_probabilities,get_fullheatmap_from_fold
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
import argparse
import shutil


__author__ = "Yudong Zhang"


def save_args_to_file(args, path):
    with open(path, 'a+') as file:
        for arg, value in vars(args).items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            file.write(f"{arg}: {value}\n")
        file.write('--------------------------\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # model
    parser.add_argument('--model_mode', choices=['deepBlink', 'DetNet', 'superpoint', 'PointDet'], default='deepBlink')
    
    # dataset
    parser.add_argument('--test_datapath', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/testdataset/test_MICROTUBULE/SNR7/')
    parser.add_argument('--datatype', choices=['8bit', '16bit'], default='8bit')

    # optimizer
    parser.add_argument('--gpu_list', nargs='+', default=[2])

    # If resume
    parser.add_argument('--ckpt_path', type=str, required=True)

    # threshold
    parser.add_argument('--thre', type=float, default=0.5)

    # Only for DetNet
    parser.add_argument('--alpha', type=float, default=0.1)

    # Only for PointDet
    parser.add_argument('--cfg', type=str,default='./config/inference_demo_coco.yaml')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    # Log and save
    parser.add_argument('--log_root', type=str, default='./Log/')
    parser.add_argument('--exp_name', type=str, default='MICROTUBULE_SNR7_deepBlink')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    opt = parse_args()
    # model
    model_mode = opt.model_mode
    
    # dataset
    test_datapath = opt.test_datapath
    datatype = opt.datatype

    # optimizer
    gpu_list = opt.gpu_list

    # if resume
    ckp_path = opt.ckpt_path

    # log and save
    Log_path = opt.log_root

    now = int(round(time.time()*1000))
    nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))
    expname = nowname+'_'+opt.exp_name+'_eval'
    # Makedirs and Save files
    if not os.path.exists(Log_path+expname):
        os.makedirs(Log_path+expname)
    
    # save this file
    thisfilepath = os.path.abspath(__file__)
    shutil.copy(thisfilepath, Log_path+expname+'/onlypred_code.py')
    # record log
    logtxt_path = Log_path+expname+'/log.txt'
    logtxt = open(logtxt_path,'a+')
    logtxt.write('\n\n')
    logtxt.write('===============Only pred===============\n')
    logtxt.write('==============={}===============\n'.format(expname))
    logtxt.close()
    save_args_to_file(opt, logtxt_path)


    # load data model
    if datatype == '16bit':
        dataloader_ins_test = func_getdataloader_pred_16(model_mode, test_datapath,batch_size=1,shuffle=False,num_workers=16,training=False)
    else:
        dataloader_ins_test = func_getdataloader_pred(model_mode, test_datapath,batch_size=1,shuffle=False,num_workers=16,training=False)

    model_ins = func_getnetwork(model_mode,opt)

    # if use GPU
    device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
    model_ins.to(device)
    if torch.cuda.device_count() > 1 and len(gpu_list) > 1:
        model_ins = nn.DataParallel(model_ins, device_ids=gpu_list)

    # load checkpoint
    c_checkpoint = torch.load(ckp_path,map_location="cuda:{}".format(gpu_list[0]))
    model_ins.load_state_dict(c_checkpoint["model_state_dict"])
    print("==> Loaded pretrianed model checkpoint '{}'.".format(ckp_path))

    thre_max = opt.thre
    # start inferance
    print(f'====>>>Threshold{thre_max}')
    model_ins.eval()
    
    
    print('===>Start prediction')

    # make save folder
    inf_dir = Log_path+expname+'/prediction_'+str(thre_max)
    if not os.path.exists(inf_dir):
        os.makedirs(inf_dir)

    for data in dataloader_ins_test:
        inp = data[0].to(device)
        name = data[1][0]
        inputimage = data[2][0].numpy()
        if model_mode == 'superpoint':

            pred = model_ins(inp)
            pred_heatmap = get_fullheatmap_from_fold(pred)[0].detach().cpu().numpy()
            pred_coords = get_coordinates(pred_heatmap,thre=thre_max)
        elif model_mode == 'DetNet':
            pred = model_ins(inp)[0].permute(1,2,0).detach().cpu().numpy()
            pred_coords = get_coordinates(pred,thre=thre_max)
        elif model_mode == 'deepBlink':
            pred = model_ins(inp)[0].permute(1,2,0).detach().cpu().numpy()
            pred_coords = get_coordinate_list(pred, image_size=max(inp.shape), probability=thre_max)
        elif model_mode == 'PointDet':
            pheatmap,poffset,psegment = model_ins(inp)
            psegment = psegment[0,0,:,:].detach().cpu().numpy()

            pred_coords = get_coordinates(psegment,thre_max)

        pred_coords_pd = pd.DataFrame(pred_coords,columns=['pos_y','pos_x'])
        pred_coords_pd.to_csv(inf_dir+'/'+name+'.csv',index = None)

        inputimage = func_normlize(inputimage,mode='maxmin_norm')
        inputimage = np.clip(np.round(inputimage*255),0,255).astype(np.uint8)

        if model_mode == 'PointDet':
            for y,x in pred_coords:
                cv2.circle(inputimage, (int(y), int(x)), 5, (0, 255, 255), 1)
        else:
            for x,y in pred_coords:
                cv2.circle(inputimage, (int(y), int(x)), 5, (0, 255, 255), 1)

        # print('save:'+name)
        cv2.imwrite(inf_dir+'/'+name+'.png',inputimage)

    message = '==>>[TEST]threthold:{:.1f}'.format(thre_max)
    print(message)
    logtxt_ = open(logtxt_path,'a+')
    logtxt_.write(message)
    logtxt_.close()
    # print('=====>>>>'+nowname)
    # print(test_datapath.split('/')[-3])
    # print(test_datapath.split('/')[-2])
    # print(test_datapath.split('/')[-1])
