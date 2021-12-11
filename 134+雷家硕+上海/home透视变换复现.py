import cv2
import numpy as np

img = cv2.imread('photo1.jpg')#透视变换不需要转灰度
result3 = img.copy()

#通过“寻找顶点”知道原图像顶点为
src = np.float32([[207,151],
       [16,603],
       [344,732],
       [518,283]])
dst = np.float32([[10,10],
       [10,400],
       [300,400],
       [300,10]])
matrix = cv2.getPerspectiveTransform(src, dst)
print('透视变换矩阵：', matrix)
result = cv2.warpPerspective(result3, matrix, (300,400))
                                                #输入的尺寸是图纸的大小，并不是图片的大小
                                                #图片的大小由dst的尺寸决定
cv2.imshow('原图像',img)
cv2.imshow('透视后',result)
cv2.waitKey()