import cv2
import numpy as np
import random
def peppersaltnoise(snr,img):
    '''
    :param snr: [0,1]
    :param img:
    :return: 单个椒盐
    '''
    noiseimg = img
    sp = noiseimg.shape[0]*noiseimg.shape[1]
    np = int(sp*snr)
    for i in range(np):
        randx = random.randint(0,noiseimg.shape[0]-1)
        randy = random.randint(0,noiseimg.shape[1]-1)
        if noiseimg[randx,randy] > 128:
            noiseimg[randx, randy] = 255
        else:
            noiseimg[randx, randy] = 0
    return noiseimg
img = cv2.imread('lenna.png',0)
img1 = peppersaltnoise(0.6, img)

img_ = cv2.imread('lenna.png',0)#因为噪声命令会覆盖原图，所以要重新加载一遍
cv2.imshow('len',img_)
cv2.imshow('pesalen',img1)
cv2.waitKey()

