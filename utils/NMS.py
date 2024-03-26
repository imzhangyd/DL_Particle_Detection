from operator import index, ne
from platform import release
import numpy as np
import pandas as pd


def nms_bbox(dets, thresh): 
    """Pure Python NMS baseline.""" 
    #dets某个类的框，x1、y1、x2、y2、以及置信度score
    #eg:dets为[[x1,y1,x2,y2,score],[x1,y1,y2,score]……]]
    # thresh是IoU的阈值     
    x1 = dets[:, 0] 
    y1 = dets[:, 1]
    x2 = dets[:, 2] 
    y2 = dets[:, 3] 
    scores = dets[:, 4] 
    #每一个检测框的面积 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    #按照score置信度降序排序 
    order = scores.argsort()[::-1] 
    keep = [] #保留的结果框集合 
    while order.size > 0: 
        i = order[0] 
        keep.append(i) #保留该类剩余box中得分最高的一个 
        #得到相交区域,左上及右下 
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        yy1 = np.maximum(y1[i], y1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]]) 
        yy2 = np.minimum(y2[i], y2[order[1:]]) 
        #计算相交的面积,不重叠时面积为0 
        w = np.maximum(0.0, xx2 - xx1 + 1) 
        h = np.maximum(0.0, yy2 - yy1 + 1) 
        inter = w * h 
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积） 
        ovr = inter / (areas[i] + areas[order[1:]] - inter) 
        #保留IoU小于阈值的box 
        inds = np.where(ovr <= thresh)[0] 
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位 
    return keep


def nms_point(points, scores, thresh):
    """执行非极大值抑制操作，用于点检测结果

    参数：
    points：形状为 (N, 2) 的numpy数组，表示N个点的二维坐标，每行包含2个值：[x, y]
    scores：形状为 (N,) 的numpy数组，表示N个点的得分
    thresh：阈值，用于控制重叠的点被抑制的程度

    返回：
    keep：一个列表，包含保留的点的索引
    """
    x = points[:, 0]  # 提取点的横坐标
    y = points[:, 1]  # 提取点的纵坐标

    areas = np.ones_like(scores)  # 由于是点检测，将点视为单位面积的矩形框，面积为1
    order = scores.argsort()[::-1]  # 根据得分对点进行降序排列，返回排列后的索引

    keep = []  # 用于保存保留的点的索引
    while order.size > 0:
        i = order[0]  # 取得分最高的点的索引
        keep.append(i)  # 将该点的索引加入保留列表

        # 计算当前点与其他点的距离
        dist_w = np.abs(x[order[1:]] - x[i])
        dist_h = np.abs(y[order[1:]] - y[i])

        dist = dist_w**2 + dist_h**2  # 计算距离的平方

        inds = np.where(dist >= thresh)[0]  # 找到距离平方大于等于阈值的点的索引
        order = order[inds + 1]  # 更新待处理的点的索引列表，去掉与当前点重叠面积较大的点

    return keep  # 返回保留的点的索引列表


if __name__ =='__main__':
    detresultpath = './20230508_exp_finetune_detector/SP_FC_1C_Control/det_result/20230508_20_27_55/dpblink_0_5/SP_FC_1C_Control_02.csv'
    detdf = pd.read_csv(detresultpath,header=0, index_col=0)
    newdf = pd.DataFrame(columns=detdf.columns)
    for fr in range(int(detdf['frame'].values.max())+1):
        thisframedet = detdf[detdf['frame'] == fr]
        points = thisframedet[['pos_x','pos_y']]
        scores = np.array([1.]*len(points))
        thresh = 25
        newpoints_index = nms_point(points.values, scores, thresh)
        newpoints = thisframedet.iloc[newpoints_index]
        newdf = newdf.append(newpoints)
    
    newdf.reset_index(drop=True, inplace=True)
    newdf.to_csv('./20230508_exp_finetune_detector/SP_FC_1C_Control/det_result/20230508_20_27_55/dpblink_0_5/SP_FC_1C_Control_02_nms.csv')
    

        