from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#灰度化
#首先读取图像，注意cv读进来的通道排列顺序是BGR，先转以后供算法用。这次不需要，直接用笨公式不需要管绝对的顺序
# img = cv2.imread('lenna.png')
# #转出来的灰度图是另外一张图像，所以先构造一张和原图像一样大（维度相同）的空白图像（矩阵），再把各像素点的灰度填进去就好了
# h , w = img.shape[:2]   #获取原图的高宽，因为每个像素也是由三维矩阵组成的，
#                         # 所以说整个图像矩阵其实是一个张量，所以shape的结果的三维而不是只有长宽
#                         #如果是RGB模式shape就是三维且顺序为1.行数（高），2.列数（宽），3.每个张量的维度（RGB）
#                         #所以这里取前两个
#                         #具体例子可见算草纸中的解密shape函数
# img_gray = np.zeros([h,w], img.dtype)   #现在单通道的灰度图每个像素只有一维，
#                                         # 那dtype是规定这个一唯的数值类型要和原图img一样
# #开始读取每一个RGB像素的矩阵值，并用浮点算法转化为灰度，再放到空格子里
# for i in range(h):  #行
#     for j in range(w):  #列
#         m = img[i, j]  # 可以直接[i,j]这样取，注意m的顺序为BGR
#         img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
#         '''
#         取int不会失真吗？这里只是一个示例
#         '''
# # print(img_gray) #就是一个张量，不会有图像
# # cv2.imshow("image show gray", img_gray)
# # cv2.waitKey()   #加这个命令就不会一闪而过了
# # plt.subplot(221)
# # img = plt.imread("lenna.png")
# # plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img_gray, cmap = 'gray')
#
# plt.show()
# print("---image lenna----")
# print(img)

def rgb_gray_diy(path): #写成定义笨方法
    img = cv2.imread(path)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], dtype= np.uint8)
    for i in range(h):
        for j in range(w):
            element_array = img[i,j]    #顺序是BGR
            img_gray[i, j] = int(element_array[0] * 0.11 +
                                 element_array[1] * 0.59 +
                                 element_array[2] * 0.3)    #手动转灰度不会归一化
    img_gray = img_gray/255 #手动归一化
    return img_gray
    #得到的是灰度图的矩阵

img_gray_raw_array = rgb_gray_diy('lenna.png')
cv2.imshow('image show raw gray', img_gray_raw_array)   #第一个必须是命名，不能省
cv2.waitKey()

# def rgb_gray(path): #写成定义掉包
#     img = cv2.imread(path)
#     img_gray = rgb2gray(img)
#     return img_gray
# img_gray_array = rgb_gray('lenna.png')
# cv2.imshow('image show gray', img_gray_array)
# cv2.waitKey()

#二值化
# def rgb_binary(path):
#     img = cv2.imread(path)
#     img_gray = rgb2gray(img)    #这里由包得到的灰度图已经归一化了，不用再除255了
#     h, w = img_gray.shape   #傻瓜，这就不用前两个了呀，直接就是二维了
#     img_binary = np.zeros([h, w])
#     for i in range(h):
#         for j in range(w):
#             pix = img_gray[i, j]
#             if pix <= 0.5:
#                 img_binary[i, j] = 0
#             else:
#                 img_binary[i, j] = 1
#     return img_binary
#
# img_binary = rgb_binary('lenna.png')
# cv2.imshow('image img_binary', img_binary)
# cv2.waitKey()