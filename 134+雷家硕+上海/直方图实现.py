import cv2
import matplotlib.pyplot as plt

#灰度图直方图
img = cv2.imread('lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#就用这个方法了
hist = cv2.calcHist([img],[0],None,[256],[0,256])#每个变量都要用[]括起来
#最简单的可视化流程
plt.figure()#新建一个图像
plt.plot(hist)
plt.show()

#彩色图像直方图,就是把各通道分开
#用calcHist函数，各参数含义为：
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数，级数的意思就是最大的灰度值取多少
ranges：横轴范围
'''
img_color = cv2.imread('lenna.png')
channels = cv2.split(img_color) #用split函数就能简单得到各通道的图像
                                #channels竟然是包含三个矩阵的元组！
colors = ('b', 'g', 'r')    #是元组
# print(type(colors))
for (channel, color) in zip(channels, colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0,256])#这里用到元组中的channel
    plt.plot(hist,color = color)#这里用到元组中的color
    plt.xlim([0,256])
    plt.show()  #随着循环分别显示
# plt.show()  #放一起显示