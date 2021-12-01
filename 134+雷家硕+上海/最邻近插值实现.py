#图像的放大和缩小，最邻近插值法
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
def nearest(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #转
    h, w, channel = img.shape   #不管几通道的图像都能用
    emptyimg = np.zeros((800, 800, channel), np.uint8)
        #uint8是专门用于存储各种图像的数据格式（包括RGB，灰度图像等），范围是从0–255
        #我不管你原图形是大于800还是小于800
    sh = 800/h  #放大倍数
    sw = 800/w
    for i in range(800):
        for j in range(800):
            x = int(i/sh)
            y = int(j/sw)   #找最邻近的时候x,y分别找，
                            #取整会四舍五入，不会直接砍掉小数部分
            emptyimg[i,j] = img[x, y]
    return emptyimg
    #如果一开始没转，那么得到的结果也是没转的
    #但plt认的是正常的RGB顺序，所以如果不转就会串色，但如果用cv显示就不需要转，反正它们是一个系列的
    #所以为了方便，在一开始就转
# img = cv2.imread('lenna.png')
img_nearest = nearest('lenna.png')
# cv2.imshow('image nearest', img_nearest)
# cv2.waitKey()
img = plt.imread('lenna.png')
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(img_nearest)
'''
中间没有涉及到RGB还是BGR的顺序问题，为什么还是颜色不对
'''
plt.show()
