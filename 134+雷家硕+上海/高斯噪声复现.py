import cv2
import numpy as np
import random
def gaussiannoise(src,means,sigma,percentage):
    noiseimg = src
    noisenum = int(percentage*noiseimg.shape[0]*noiseimg.shape[1])
    for i in range(noisenum):
        randx = random.randint(0,noiseimg.shape[0]-1)
        randy = random.randint(0,noiseimg.shape[1]-1)
        noiseimg[randx,randy] = noiseimg[randx,randy] + random.gauss(means, sigma)
        if noiseimg[randx,randy] < 0:
            noiseimg[randx, randy] = 0
        elif noiseimg[randx,randy] > 255:
            noiseimg[randx, randy] = 255
    return noiseimg
img = cv2.imread('lenna.png',0)#高斯噪声只能处理一个通道或灰度图
img1 = gaussiannoise(img, 2, 4, 0.8)
cv2.imshow('len',img)
cv2.imshow('gausslen',img1)
cv2.waitKey()

