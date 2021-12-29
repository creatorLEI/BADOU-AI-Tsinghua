import cv2
from skimage import util
img = cv2.imread('lenna.png')#这种噪声可以直接处理三通道
noise = util.random_noise(img, mode='poisson')
cv2.imshow('len',img)
cv2.imshow('noi',noise)
cv2.waitKey()
