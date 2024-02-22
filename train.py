from optimizer.choose_optimizer import func_getoptimizer
from dataset.dataload import func_getdataloader
from loss.choose_loss import func_getloss
from model.choose_net import func_getnetwork
from optimizer.update_opti_lr import func_update_opti_lr
from model.init_model import init_weights
from visdom import Visdom
import torch
import torch.nn as nn
import numpy as np
import os
import time
now = int(round(time.time()*1000))
nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))


operation = 'train' # train
# txtpath = './Data/20211209train_1.txt'
datapath = './Data/'
total_epoch = 3
bs = 2
model_mode = 'unet'
loss_mode = 'iou'
opti_mode = 'SGD'
lr = 0.1
decay_every = 4
gpu_list = [0,1,2,3]


viz = Visdom(env=nowname, port=4004)
# record log
logtxt_path = './Log/log.txt'
logtxt = open(logtxt_path,'a+')
logtxt.write('\n\n')
logtxt.write('=============={}===============\n'.format(nowname))
logtxt.write('operation={}\n'.format(operation))
logtxt.write('datapath={}\n'.format(datapath))
logtxt.write('batchsize={}\n'.format(bs))
logtxt.write('total_epoch={}\n'.format(total_epoch))
logtxt.write('model_mode={}\n'.format(model_mode))
logtxt.write('loss_mode={}\n'.format(loss_mode))
logtxt.write('lr={}\n'.format(lr))
logtxt.write('decay_every={}\n'.format(decay_every))
logtxt.write('--------------------------------\n')
logtxt.close()


# load data model loss and optimizer 
dataloader_ins = func_getdataloader(datapath+'train',batch_size=bs,shuffle=True,num_workers=0)
model_ins = func_getnetwork(model_mode)
init_weights(model_ins)
cal_loss_ins = func_getloss(loss_mode)
modelparams_list = [{'params': model_ins.parameters()}]
optimizer_ins = func_getoptimizer(modelparams_list, opti_mode, lr=lr, momentum=0.9, wd=0.0005)

# if use GPU
device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
model_ins.to(device) # 移动模型到cuda
if torch.cuda.device_count() > 1 and len(gpu_list) > 1:
    model_ins = nn.DataParallel(model_ins, device_ids=gpu_list) # 包装为并行风格模型

# make save folder
ckt_dir = './Log/'+nowname+'/checkpoints'
if not os.path.exists(ckt_dir):
    os.makedirs(ckt_dir)

# start train
step = 0
loss_ = 0
for epoch in range(total_epoch):
    epochloss_list = []
    since = time.time()
    for data in dataloader_ins:
        step +=1
        inp = data[0].to(device)
        lab = data[1].to(device)

        pred = model_ins(inp)
        loss = cal_loss_ins(pred,lab)

        optimizer_ins.zero_grad()
        loss.backward()
        optimizer_ins.step()
        lr = func_update_opti_lr(optimizer_ins,epoch,decay_every)

        # print('epoch:{} step:{} lr:{:.7f} loss:{:5f}'.format(epoch+1,step,lr,loss))
        # visualize step train loss
        loss_ = loss.item()
        viz.line(Y=[loss_], X=torch.Tensor([step]), win='train step loss', update='append', opts=dict(title="Training StepLoss", xlabel="Step", ylabel="Loss"))
        epochloss_list.append(loss_)

    # visualize epoch train loss
    viz.line(Y=[np.array(epochloss_list).mean()], X=torch.Tensor([epoch + 1]), win='train epoch loss', update='append', opts=dict(title="Training EpochLoss", xlabel="epoch", ylabel="Loss"))
    # save every model
    torch.save({'epoch': epoch+1,
                'model_state_dict': model_ins.state_dict(),
                'optimizer': optimizer_ins.state_dict(),
                'loss': np.array(epochloss_list).mean()},
                os.path.join(ckt_dir, "checkpoints_" + str(epoch + 1) + ".pth"))
    # claculate time
    time_elapsed = time.time() - since
    # record loss time 
    message = 'epoch:{} lr:{:.7f} loss:{:5f} elapse:{:.0f}m {:.0f}s'.format(epoch+1, lr,np.array(epochloss_list).mean(), time_elapsed // 60, time_elapsed % 60)
    print(message)
    logtxt = open(logtxt_path,'a+')
    logtxt.write(message+'\n')
    logtxt.close()