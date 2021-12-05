#除了调包，还要自己编公式
#调包
import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''
#灰度图像
# img = cv2.imread('lenna.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist(img, [0], None, [256], [0,256])#均衡化前直方图
# # plt.figure()
# # plt.plot(hist)
# # plt.show()
# dst = cv2.equalizeHist(img, None)#均衡化后的图像
# hist1 = cv2.calcHist(dst, [0], None, [256], [0,256])
# plt.figure()
# plt.plot(hist1)
# plt.show()

#彩色图像
# img1 = cv2.imread('lenna.png')
# channels = cv2.split(img1)
# colors = ('b', 'g', 'r')
# for (channel, color) in zip(channels, colors):
#     dst = cv2.equalizeHist(channel, None)
#     hist = cv2.calcHist(channel, [0], None, [256], [0,256])
#     hist1 = cv2.calcHist(dst, [0], None, [256], [0, 256])
#     plt.subplot(2, 1, 1)
#     plt.plot(hist, color=color)  # 这里用到元组中的color
#     plt.subplot(2,1,2)
#     plt.plot(hist1, color = color)
#     plt.xlim([0, 256])
# plt.show()



# def hist_dis(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     h, w = img.shape
#     hist = dict(zip(range(256), [0]*256))
#     for i in range(h):
#         for j in range(w):
#             pxl = img[i, j]
#             hist[pxl] += 1
#     return hist
#     #得到图像像素值直方图分布字典
#
#
# def accu(hist):
#     accu_dict = {}
#     for i in hist.keys():
#         accu_dict[i] = sum(hist[a] for a in range(i+1))
#     return accu_dict
#
#
# def equ(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_hist = hist_dis(path)
#     accu_hist = accu(img_hist)
#     h, w = img.shape
#     new = np.zeros([h, w])
#     for i in range(h):
#         for j in range(w):
#             key = img[i, j]
#             sum_hist = accu_hist[key]
#             new[i, j] = int(sum_hist*256/(h*w) - 1)
#     return new
#
#自己编公式
class diy_equ(object):
    def __init__(self, img):#需要的唯一一个外部变量就是img
        self.img = img

        #每个定义得到的结果也要self，因为可能会类内引用
        self.hist = self.hist_dis()
        self.accu_dict = self.accu()


    def hist_dis(self):
        h, w = self.img.shape
        hist = dict(zip(range(256), [0] * 256))
        for i in range(h):
            for j in range(w):
                pxl = self.img[i, j]
                hist[pxl] += 1
        return hist
        # 得到图像像素值直方图分布字典

    def accu(self):
        accu_dict = {}
        '''如何调用class里面的函数?
        在self里面加def的结果
        '''
        for i in self.hist.keys():
            accu_dict[i] = sum(self.hist[a] for a in range(i + 1))
        return accu_dict

    def equ(self):
        h, w = self.img.shape
        new = np.zeros([h, w])
        for i in range(h):
            for j in range(w):
                key = self.img[i, j]
                sum_hist = self.accu_dict[key]
                new[i, j] = int(sum_hist * 256 / (h * w) - 1)
        print('new', new)
        return new

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diy = diy_equ(img)
