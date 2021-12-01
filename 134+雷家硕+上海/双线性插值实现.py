import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# def bilinear(path):
#     img = cv2.imread(path)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     src_h, src_w, channel = img.shape       #src是原矩阵的维度
#     # dst_h, dst_w = out_dim[1], out_dim[0]   #这个是新矩阵的维度
#     #                                         #因为在矩阵的维度里面高h是第二维，所以dst_w是out_dim[0]
#     #                                         #而且x对应的是w
#     emptyimg = np.zeros((800, 800, channel), dtype=np.uint8)
#                                             #设立一个新矩阵
#     # scale_x, scale_y = float(src_w/dst_w), float(src_h/dst_h)
#                                             #放大的倍数的倒数
#     for c in range(channel):   #每一个通道都要分别做插值，不能整个矩阵一起算
#         for i in range(800):
#             for j in range(800):
#                 x  = float(i * src_h / 800)#x,y是原图坐标
#                 y  = float(j * src_w / 800)
#                 x1 = int(np.floor(x))
#                 x2 = min(x1+1, src_w-1)    #上限要有一个限制
#                 y1 = int(np.floor(y))
#                 y2 = min(y1+1, src_h-1)
#                 f_x_y1 = (x - x1)*img[x2, y1, c] + (x2 - x)*img[x1, y1, c]
#                         #x2-x1肯定是1，所以就不用再除了
#                 f_x_y2 = (x - x1)*img[x2, y2, c] + (x2 - x)*img[x1, y2, c]
#                 f_x_y = (y - y1)*f_x_y2 + (y2 - y)*f_x_y1
#                         #同理y2-y1
#                 emptyimg[i, j, c] = f_x_y
#     return emptyimg
#
# bilinearimg = bilinear('lenna.png')
# cv2.imshow('image bilinear', bilinearimg)
# cv2.waitKey()

def bilinear(path):
    img = cv2.imread(path)
    # 获取原图的维度，这是最邻近和双线性插值都要有的步骤
    h, w, channel = img.shape
    # 构造空白矩阵，两种方法都要有
    emptyimg = np.zeros((800, 800, channel), dtype=np.uint8)
                                                            #因为是插值三通道的原图，不是灰度图，所以要构造三维的空白矩阵
                                                            #这个dtype方式通用
    #这里开始不一样，最邻近算法不分通道，直接暴力把整个img[i, j]放上去
    #但双线性是做插值，得到的是值，所以必须每个通道都要分别讨论
    for c in range(channel):
        #这里两种方法都是循环新矩阵
        for i in range(800):
            for j in range(800):
                x = float(i * w/800)
                y = float(j * h/800)
                x1 = int(np.floor(x))
                x2 = min(x1+1, w-1) #x,y是原图像的坐标
                y1 = int(np.floor(y))
                y2 = min(y1+1, h-1)
                f_x_y1 = (x - x1)*img[x2, y1, c] + (x2 - x)*img[x1, y1, c]
                f_x_y2 = (x - x1)*img[x2, y2, c] + (x2 - x)*img[x1, y2, c]
                f_x_y = (y - y1)*f_x_y2 + (y2 - y)*f_x_y1
                emptyimg[i, j, c] = f_x_y
    return emptyimg

bilinearimg = bilinear('lenna.png')
cv2.imshow('image bilinear', bilinearimg)
cv2.waitKey()

