from optimizer.choose_optimizer import func_getoptimizer
from dataset.dataload import func_getdataloader
from loss.choose_loss import func_getloss
from model.choose_net import func_getnetwork
from optimizer.update_opti_lr import func_update_opti_lr
from model.init_model import init_weights
from creterion.iou import func_ioucreterion
from loss.detnetloss import f1_score
# from creterion.rmse import z
# from creterion.rmse import rmse_
from visdom import Visdom
import torch
import torch.nn as nn
import numpy as np
import os
import time
# import ipdb
from config.default import _C as cfg
from config.default import update_config
import argparse
# import matplotlib.pyplot as plt
from loss.detnetloss import EarlyStopping

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


if __name__ == "__main__":

    # for scene in ['RECEPTOR','MICROTUBULE','VESICLE']:
    scene = 'RECEPTOR'
    snr_ = '7'
    now = int(round(time.time()*1000))
    nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))

    # ====================================================================

    args = parse_args()
    update_config(cfg, args)

    operation = 'trainval' # train trainval inference
    train_datapath = '/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/train_'+scene+'/SNR'+snr_+'/'
    val_datapath = '/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/val_'+scene+'/SNR'+snr_+'/'
    total_epoch = 200

    model_mode = 'PointDet'
    loss_mode = 'heatmap_offset'
    opti_mode = 'amsAdam'
    lr = 0.001
    decay_every = 10e4
    gpu_list = [3]
    # bs = len(gpu_list)*2
    bs = 2
    ckp_name = '20220708_03_23_57'
    # ckp_name = '20211215_20_13_10' # None or path
    if not ckp_name is None:
        load_epoch = 10
        ckp_path = os.path.join('./Log/',ckp_name+ "/checkpoints/checkpoints_" + str(load_epoch) + ".pth")

    # ====================================================================

    logtxt_path = './Log/'+nowname+'/log.txt'
    if not os.path.exists('./Log/'+nowname):
        os.makedirs('./Log/'+nowname)

    viz = Visdom(env=nowname, port=2345)
    # record log
    # logtxt_path = './Log/log.txt'
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
    dataloader_ins = func_getdataloader(train_datapath,batch_size=bs,shuffle=True,num_workers=16,training=True)
    dataloaderval_ins = func_getdataloader(val_datapath,batch_size=1,shuffle=True,num_workers=16,training=True)
    model_ins = func_getnetwork(model_mode,cfg)
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
    # cal_loss_ins = func_getloss(loss_mode,1,cfg)
    cal_loss_ins2 = func_getloss('soft_dice',1,cfg)
    modelparams_list = [{'params': model_ins.parameters()}]
    optimizer_ins = func_getoptimizer(modelparams_list, opti_mode, lr=lr, momentum=0.9, wd=0.0005)


    # make save folder
    ckt_dir = './Log/'+nowname+'/checkpoints'
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
        epochheatmaploss_list = []
        epochoffsetloss_list = []
        epochsegmentloss_list = []
        since = time.time()
        for data in dataloader_ins:
            # ipdb.set_trace()
            step +=1
            inp = data[0].to(device)
            lab = data[1].to(device)

            heatmap = data[2].to(device)
            mask = data[3].to(device)

            offset = data[4].to(device)
            offset_w = data[5].to(device)
                    

            pheatmap,poffset,psegment = model_ins(inp)
            # heatmap_loss, offset_loss = cal_loss_ins(pheatmap,poffset, heatmap, mask, offset, offset_w) #output, poffset, heatmap, mask, offset, offset_w)
            seg_loss = cal_loss_ins2(psegment,lab)
            # print(heatmap_loss)
            # print(offset_loss)
            # print(seg_loss)

            # loss = heatmap_loss+offset_loss
            loss = seg_loss
            optimizer_ins.zero_grad()
            loss.backward()
            optimizer_ins.step()
            lr = func_update_opti_lr(optimizer_ins,epoch,decay_every)


            # epochheatmaploss_list.append(heatmap_loss.item())
            # epochoffsetloss_list.append(offset_loss.item())
            # epochsegmentloss_list.append(seg_loss.item())

            loss_ = loss.item()
            epochloss_list.append(loss_)

        # visualize epoch 3loss
        # viz.line(Y=[np.array(epochheatmaploss_list).mean()], X=torch.Tensor([epoch]), win='train epoch  heatmaploss', update='append', opts=dict(title="Training Heatmap EpochLoss", xlabel="epoch", ylabel="Loss"))
        # viz.line(Y=[np.array(epochoffsetloss_list).mean()], X=torch.Tensor([epoch]), win='train epoch offsetloss', update='append', opts=dict(title="Training Offset EpochLoss", xlabel="epoch", ylabel="Loss"))
        # viz.line(Y=[np.array(epochsegmentloss_list).mean()], X=torch.Tensor([epoch]), win='train epoch segmentloss', update='append', opts=dict(title="Training Segment EpochLoss", xlabel="epoch", ylabel="Loss"))

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

                heatmap = data[2].to(device)
                mask = data[3].to(device)

                offset = data[4].to(device)
                offset_w = data[5].to(device)
                image_ori = data[-1]
                # lab = data[1]
                num_ = len(lab)
                t_num += num_
                pheatmap,poffset,psegment = model_ins(inp)
                # if epochloss_t == 0:
                #     viz.image(image_ori[:,:,0]/255.0,win='image',opts=dict(title="image", caption='image'))
                #     viz.image(data[2][0,0,:,:].numpy(),win='heatmap',opts=dict(title="heatmap", caption='heatmap'))
                #     viz.image(data[1][0,0,:,:].numpy(),win='segment',opts=dict(title='segment',caption='segment'))
                #     viz.image(data[3][0,0,:,:].numpy(),win='heatmap_mask',opts=dict(title='heatmap_mask',caption='heatmap_mask'))
                #     viz.image(pheatmap.detach().cpu().numpy()[0,0,:,:],win='pred_heatmap',opts=dict(title='pred_heatmap',caption='pred_heatmap'))
                #     viz.image(poffset.detach().cpu().numpy()[0,0,:,:],win='pred_offset',opts=dict(title='pred_offset',caption='pred_offset'))
                #     viz.image(psegment.detach().cpu().numpy()[0,0,:,:],win='pred_psegment',opts=dict(title='pred_segment',caption='pred_segment'))

                # heatmap_loss, offset_loss = cal_loss_ins(pheatmap,poffset, heatmap, mask, offset, offset_w) #output, poffset, heatmap, mask, offset, offset_w)
                seg_loss = cal_loss_ins2(psegment,lab)

                loss = seg_loss
                # loss = heatmap_loss+offset_loss
                                
                iouval = f1_score(psegment.cpu(),lab.detach().cpu())
                loss_ = loss.item()
                epochloss_t += loss_*num_
                epochiou_t += np.array(iouval).sum()
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


    mesg = '==>best epoch:{} val loss:{:5f}'.format(best_epoch,best_valloss)
    print(mesg)
    logtxt = open(logtxt_path,'a+')
    logtxt.write(mesg+'\n')
    logtxt.close()
