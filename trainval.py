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

from loss.detnetloss import EarlyStopping
# import ipdb
import shutil

if __name__ == "__main__":

    dataroot = '/data/ldap_shared/synology_shared/zyd/data/'

    scen = 'RECEPTOR'
    sn = 'SNR2'
    Log_path = './Log/'
    logtxt_path_opt = Log_path+scen+'_'+sn+'/log.txt'
    if not os.path.exists(Log_path+scen+'_'+sn):
        os.makedirs(Log_path+scen+'_'+sn)
    # for alpha in np.linspace(0,1,21):
    opt = {}
    opt['alpha'] = 0.1

    now = int(round(time.time()*1000))
    nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))

    # ====================================================================
    operation = 'trainval' # train trainval inference
    # train_datapath = '/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/train_'+scen+'/'+sn+'/'
    # val_datapath = '/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/val_'+scen+'/'+sn+'/'
    train_datapath = '/mnt/data1/ZYDdata/helabdata_train_detection/SP_FC_1C_Control/trainvaltest/train/'
    val_datapath = '/mnt/data1/ZYDdata/helabdata_train_detection/SP_FC_1C_Control/trainvaltest/val/'
    
    total_epoch = 200
    datatype = '16bit' #

    model_mode = 'deepBlink'

    if model_mode == 'deepBlink':
        from loss.deepblinkloss import f1_score
    else:
        from loss.detnetloss import f1_score
    loss_mode = 'combined_dice_rmse' #combined_dice_rmse  soft_dice
    opti_mode = 'amsAdam'
    lr = 0.001
    decay_every = 1e10
    gpu_list = [0]
    # bs = len(gpu_list)*2
    bs = 2
    ckp_name = None
    # ckp_name = '20211215_20_13_10' # None or path
    if not ckp_name is None:
        load_epoch = 28
        ckp_path = os.path.join(Log_path,ckp_name+ "/checkpoints/checkpoints_" + str(load_epoch) + ".pth")

    # ====================================================================
    
    logtxt_path = Log_path+nowname+'/log.txt'
    if not os.path.exists(Log_path+nowname):
        os.makedirs(Log_path+nowname)

    thisfilepath = os.path.abspath(__file__)
    shutil.copy(thisfilepath, Log_path+nowname+'/trainval_file.py')

    viz = Visdom(env=nowname, port=4006)
    # record log
    logtxt = open(logtxt_path,'a+')
    logtxt.write('\n\n')
    logtxt.write('=============={}===============\n'.format(nowname))
    logtxt.write('operation={}\n'.format(operation))
    logtxt.write('train_datapath={}\n'.format(train_datapath))
    logtxt.write('val_datapath={}\n'.format(val_datapath))
    logtxt.write('batchsize={}\n'.format(bs))
    logtxt.write('total_epoch={}\n'.format(total_epoch))
    logtxt.write('model_mode={}\n'.format(model_mode))
    logtxt.write('loss_mode={}\n'.format(loss_mode))
    logtxt.write('lr={}\n'.format(lr))
    logtxt.write('decay_every={}\n'.format(decay_every))
    if not ckp_name is None:
        logtxt.write('ckp_path={}\n'.format(ckp_path))
    logtxt.write('--------------------------------\n')
    logtxt.close()


    # load data model loss and optimizer 
    if datatype == '16bit':
        dataloader_ins = func_getdataloader_16(model_mode,train_datapath,batch_size=bs,shuffle=True,num_workers=16)
        dataloaderval_ins = func_getdataloader_16(model_mode,val_datapath,batch_size=bs,shuffle=True,num_workers=16)  # bs=1?
    else:
        dataloader_ins = func_getdataloader(model_mode,train_datapath,batch_size=bs,shuffle=True,num_workers=16)
        dataloaderval_ins = func_getdataloader(model_mode,val_datapath,batch_size=bs,shuffle=False,num_workers=16)
    model_ins = func_getnetwork(model_mode,opt)
    init_weights(model_ins)

    # if use GPU
    device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
    model_ins.to(device) # 移动模型到cuda
    if torch.cuda.device_count() > 1 and len(gpu_list) > 1:
        model_ins = nn.DataParallel(model_ins, device_ids=gpu_list) # 包装为并行风格模型

    # if load checkpoint model
    if not ckp_name is None:
        c_checkpoint = torch.load(ckp_path)
        model_ins.load_state_dict(c_checkpoint["model_state_dict"])
        print("==> Loaded pretrianed model checkpoint '{}'.".format(ckp_path))
    cal_loss_ins = func_getloss(loss_mode)
    modelparams_list = [{'params': model_ins.parameters()}]
    optimizer_ins = func_getoptimizer(modelparams_list, opti_mode, lr=lr, momentum=0.9, wd=0.0005)


    # make save folder
    ckt_dir = Log_path+nowname+'/checkpoints'
    if not os.path.exists(ckt_dir):
        os.makedirs(ckt_dir)

    start_epoch = 1
    # load checkpoint
    if not ckp_name is None:
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

            pred = model_ins(inp)
            # if model_mode == 'unet':
            #     loss = cal_loss_ins(pred,lab)
            # if model_mode == 'unetunet':
            #     loss = cal_loss_ins(pred[0],lab)*0.5 + cal_loss_ins(pred[1],lab)*0.5
            # else:
            loss = cal_loss_ins(pred,lab)
            optimizer_ins.zero_grad()
            loss.backward()
            optimizer_ins.step()
            lr = func_update_opti_lr(optimizer_ins,epoch,decay_every)

            # print('epoch:{} step:{} lr:{:.7f} loss:{:5f}'.format(epoch+1,step,lr,loss))
            # visualize step train loss
            loss_ = loss.item()
            # viz.line(Y=[loss_], X=torch.Tensor([step]), win='train step loss', update='append', opts=dict(title="Training StepLoss", xlabel="Step", ylabel="Loss"))
            epochloss_list.append(loss_)

        # visualize epoch train loss
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
        # logtxt = open(logtxt_path,'a+')
        # logtxt.write(message+'\n')
        # logtxt.close()


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
                pred = model_ins(inp)
                # if model_mode == 'unet':
                #     loss = cal_loss_ins(pred.cpu(),lab)
                #     iouval = func_ioucreterion(pred.cpu(),lab,0.5)
                # if model_mode == 'unetunet':
                #     loss = cal_loss_ins(pred[0].cpu(),lab)*0.5 + cal_loss_ins(pred[1].cpu(),lab)*0.5
                #     iouval = np.array(func_ioucreterion(pred[0].cpu(),lab,0.5)).mean()*0.5 + np.array(func_ioucreterion(pred[1].cpu(),lab,0.5)).mean()*0.5
                # else:
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
                print("此时早停！")
                break

    mesg = '==>best epoch:{} val loss:{:.5f}'.format(best_epoch,best_valloss)
    print(mesg)
    logtxt = open(logtxt_path,'a+')
    logtxt.write(mesg+'\n')
    logtxt.close()
    if model_mode == 'DetNet':
        mmess = '{}--alpha={:.3f}--best epoch:{}--val loss:{:.5f}\n'.format(nowname,opt['alpha'],best_epoch,best_valloss)
        print(mmess)
        logtxt_ = open(logtxt_path_opt,'a+')
        logtxt_.write(mmess)
        logtxt_.close()
