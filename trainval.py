"""

Train a particle detection model, evaluate on the validation set after each epoch, 
and save checkpoints at the end of each epoch.

"""
from optimizer.choose_optimizer import func_getoptimizer
from dataset.dataload import func_getdataloader,func_getdataloader_16
from loss.choose_loss import func_getloss
from model.choose_net import func_getnetwork
from optimizer.update_opti_lr import func_update_opti_lr
from model.init_model import init_weights
from creterion.iou import func_ioucreterion
# from loss.detnetloss import f1_score
from utils.data import get_fullheatmap_from_fold
# from creterion.rmse import z
# from creterion.rmse import rmse_
from visdom import Visdom
import torch
import torch.nn as nn
import numpy as np
import os
import time
import argparse
from loss.detnetloss import EarlyStopping
# import ipdb
import shutil


__author__ = "Yudong Zhang"


# '/mnt/data1/ZYDdata/helabdata_train_detection/SP_FC_1C_Control/trainvaltest/train/'
# '/mnt/data1/ZYDdata/helabdata_train_detection/SP_FC_1C_Control/trainvaltest/val/'


def save_args_to_file(args, path):
    with open(path, 'a+') as file:
        for arg, value in vars(args).items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            file.write(f"{arg}: {value}\n")
        file.write('--------------------------')

def parse_args_():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # model
    parser.add_argument('--model_mode', choices=['deepBlink', 'DetNet', 'superpoint', 'PointDet'], default='deepBlink')
    
    # dataset
    parser.add_argument('--train_datapath', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/train_VESICLE/SNR4/')
    parser.add_argument('--val_datapath', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/val_VESICLE/SNR4/')
    parser.add_argument('--datatype', choices=['8bit', '16bit'], default='8bit')

    # optimizer
    parser.add_argument('--opti_mode', type=str, default='amsAdam')
    parser.add_argument('--lr', type=float, default=0.0001) # 0.0001 for deepBlink, 0.001 for others
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--decay_every', type=int, default=1e10)
    parser.add_argument('--loss_mode', type=str, default='combined_dice_rmse') # combined_dice_rmse for deepBlink , soft_dice for others
    parser.add_argument('--gpu_list', nargs='+', default=[2])

    # If resume
    parser.add_argument('--ckpt_path', type=str, default=None)

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
    parser.add_argument('--exp_name', type=str, default='VESICEL_SNR4_deepBlink')
    parser.add_argument('--use_visdom', type=bool, default=False)
    parser.add_argument('--port',type=int, default=4006)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    opt = parse_args_()
    # model
    model_mode = opt.model_mode
    
    # dataset
    train_datapath = opt.train_datapath
    val_datapath = opt.val_datapath
    datatype = opt.datatype

    # optimizer
    opti_mode = opt.opti_mode
    lr = opt.lr
    bs = opt.bs
    total_epoch = opt.epoch
    decay_every = opt.decay_every
    loss_mode = opt.loss_mode
    gpu_list = opt.gpu_list

    # if resume
    ckp_path = opt.ckpt_path

    # log and save
    Log_path = opt.log_root

    now = int(round(time.time()*1000))
    nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))
    expname = nowname+'_'+opt.exp_name+'_trainval'
    
    # ------------------------------------------------------------------------

    # If use visdom
    if opt.use_visdom:
        # start visdom
        viz = Visdom(env=expname, port=opt.port)

    # Makedirs and Save files
    if not os.path.exists(Log_path+expname):
        os.makedirs(Log_path+expname)
    
    # save this file
    thisfilepath = os.path.abspath(__file__)
    shutil.copy(thisfilepath, Log_path+expname+'/trainval_code.py')
    
    # record log
    logtxt_path = Log_path+expname+'/log.txt'
    logtxt = open(logtxt_path,'a+')
    logtxt.write('\n\n')
    logtxt.write('===============Trainval===============\n')
    logtxt.write('==============={}===============\n'.format(expname))
    logtxt.close()
    save_args_to_file(opt, logtxt_path)


    # load data model loss and optimizer 
    if datatype == '16bit':
        dataloader_ins = func_getdataloader_16(model_mode,train_datapath,batch_size=bs,shuffle=True,num_workers=16)
        dataloaderval_ins = func_getdataloader_16(model_mode,val_datapath,batch_size=1,shuffle=True,num_workers=16)
    else:
        dataloader_ins = func_getdataloader(model_mode,train_datapath,batch_size=bs,shuffle=True,num_workers=16)
        dataloaderval_ins = func_getdataloader(model_mode,val_datapath,batch_size=1,shuffle=False,num_workers=16)
    model_ins = func_getnetwork(model_mode,opt)
    init_weights(model_ins)

    # if use GPU
    device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
    model_ins.to(device)
    if torch.cuda.device_count() > 1 and len(gpu_list) > 1:
        model_ins = nn.DataParallel(model_ins, device_ids=gpu_list)

    # if load checkpoint model
    if not ckp_path is None:
        c_checkpoint = torch.load(ckp_path)
        model_ins.load_state_dict(c_checkpoint["model_state_dict"])
        print("==> Loaded pretrianed model checkpoint '{}'.".format(ckp_path))
    cal_loss_ins = func_getloss(loss_mode)
    modelparams_list = [{'params': model_ins.parameters()}]
    optimizer_ins = func_getoptimizer(modelparams_list, opti_mode, lr=lr, momentum=0.9, wd=0.0005)


    # make save folder
    ckt_dir = Log_path+expname+'/checkpoints'
    if not os.path.exists(ckt_dir):
        os.makedirs(ckt_dir)

    start_epoch = 1
    # load checkpoint
    if not ckp_path is None:
        start_epoch = c_checkpoint['epoch']+1


    early_stopping = EarlyStopping(patience=10,path = os.path.join(ckt_dir, "best_checkpoints.pth"))

    print('start epoch={}'.format(start_epoch))
    # start train
    step = 0
    loss_ = 0

    best_valloss = 100
    best_epoch = 1

    for epoch in range(start_epoch,total_epoch+1):
        model_ins.train()
        epochloss_list = []
        since = time.time()
        for data in dataloader_ins:
            # ipdb.set_trace()
            step +=1
            inp = data[0].to(device)
            lab = data[1].to(device)

            if model_mode == 'PointDet':
                heatmap = data[2].to(device)
                mask = data[3].to(device)

                offset = data[4].to(device)
                offset_w = data[5].to(device)
                        

                pheatmap,poffset,psegment = model_ins(inp)
                seg_loss = cal_loss_ins(psegment,lab)
                loss = seg_loss
            else:
            
                pred = model_ins(inp)
                loss = cal_loss_ins(pred,lab)
            optimizer_ins.zero_grad()
            loss.backward()
            optimizer_ins.step()
            lr = func_update_opti_lr(optimizer_ins,epoch,decay_every)

            # visualize step train loss
            loss_ = loss.item()
            epochloss_list.append(loss_)

        # visualize epoch train loss
        if opt.use_visdom:
            viz.line(Y=[np.array(epochloss_list).mean()], X=torch.Tensor([epoch]), win='train epoch loss', update='append', opts=dict(title="Training EpochLoss", xlabel="epoch", ylabel="Loss"))
        # save every model
        torch.save({'epoch': epoch,
                    'model_state_dict': model_ins.state_dict(),
                    'optimizer': optimizer_ins.state_dict(),
                    'loss': np.array(epochloss_list).mean()},
                    os.path.join(ckt_dir, "checkpoints_" + str(epoch) + ".pth"))
        # claculate time
        time_elapsed = time.time() - since
        # record loss time 
        message = 'train_epoch:{} lr:{:.7f} loss:{:5f} elapse:{:.0f}m {:.0f}s'.format(epoch, lr,np.array(epochloss_list).mean(), time_elapsed // 60, time_elapsed % 60)
        print(message)

        if model_mode == 'deepBlink':
            from loss.deepblinkloss import f1_score
        else:
            from loss.detnetloss import f1_score

        # val ---------------------------------------------------------
        with torch.no_grad():
            model_ins.eval()
            epochloss_t = 0.
            epochiou_t = 0.
            since = time.time()
            t_num = 0
            for data in dataloaderval_ins:
                inp = data[0].to(device)
                lab = data[1].to(device)
                num_ = len(lab)
                t_num += num_
                if model_mode == 'PointDet':
                    heatmap = data[2].to(device)
                    mask = data[3].to(device)

                    offset = data[4].to(device)
                    offset_w = data[5].to(device)
                    image_ori = data[-1]
                    pred = model_ins(inp)
                    pheatmap,poffset,psegment = model_ins(inp)
                    seg_loss = cal_loss_ins(psegment,lab)

                    loss = seg_loss
                    # loss = heatmap_loss+offset_loss
                                    
                    iouval = f1_score(psegment.cpu(),lab.detach().cpu())
                else:
                    pred = model_ins(inp)
                    loss = cal_loss_ins(pred,lab)
                    if model_mode == 'superpoint':
                        pred = get_fullheatmap_from_fold(pred)
                        lab = get_fullheatmap_from_fold(lab)
                    iouval = f1_score(pred,lab)
                # loss = cal_loss_ins(pred[0],lab)*0.5 + cal_loss_ins(pred[1],lab)*0.5
                loss_ = loss.item()
                f1_ = iouval.item()
                epochloss_t += loss_*num_
                # epochiou_t += np.array(iouval).sum()
                epochiou_t += f1_*num_
            if opt.use_visdom:
                # visualize epoch val loss
                viz.line(Y=[epochloss_t/t_num], X=torch.Tensor([epoch]), win='val epoch loss', update='append', opts=dict(title="Validation EpochLoss", xlabel="epoch", ylabel="Loss"))
                viz.line(Y=[epochiou_t/t_num], X=torch.Tensor([epoch]), win='val f1score', update='append', opts=dict(title="Validation F1score", xlabel="epoch", ylabel="Loss"))
            if (epochloss_t/t_num) < best_valloss:
                best_valloss = epochloss_t/t_num
                best_epoch = epoch
            
            # claculate time
            time_elapsed = time.time() - since
            # record loss time 
            message = 'Valid_epoch:{} lr:{:.7f} loss:{:5f} elapse:{:.0f}m {:.0f}s'.format(epoch, lr,epochloss_t/t_num, time_elapsed // 60, time_elapsed % 60)
            print(message)
            logtxt = open(logtxt_path,'a+')
            logtxt.write(message+'\n')
            logtxt.close()

            early_stopping(epochloss_t/t_num, model_ins)
            if early_stopping.early_stop:
                print(">>>>Early Stop")
                break

    mesg = '==>best epoch:{} val loss:{:.5f}'.format(best_epoch,best_valloss)
    print(mesg)
    logtxt = open(logtxt_path,'a+')
    logtxt.write(mesg+'\n')
    logtxt.close()

    if model_mode == 'DetNet':
        mmess = '{}--alpha={:.3f}--best epoch:{}--val loss:{:.5f}\n'.format(expname,opt['alpha'],best_epoch,best_valloss)
        print(mmess)
        logtxt_ = open(logtxt_path,'a+')
        logtxt_.write(mmess)
        logtxt_.close()
