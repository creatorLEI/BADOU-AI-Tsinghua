import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenna.png')
#不要点的位置，只留下像素值和三个通道
data = img.reshape((-1,3))
data = np.float32(data)
#给定kmeans算法的停止条件
criteria = (cv2.TERM_CRITERIA_EPS+
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#给定kmeans算法的初始中心选择
flags = cv2.KMEANS_RANDOM_CENTERS
#聚成4类和8类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None,criteria, 10, flags)
compactness8, labels8, centers8 = cv2.kmeans(data, 8, None,criteria, 10, flags)
#将图像转为uint8二维类型
centers4 = np.uint8(centers4)#摘出质心并将质心取整
res4 = centers4[labels4.flatten()]#按标签将每个点替换为取整后的质心
dst4 = res4.reshape((img.shape))#恢复成正常彩图维度

centers8 = np.uint8(centers8)
res8 = centers8[labels8.flatten()]
dst8 = res8.reshape((img.shape))
# cv2.imshow('test',dst4)
# cv2.waitKey()

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#如果想多张图画到一起，必须用plt
#而plt只认rgb，所以要先转
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst41 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst81 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
titles = [u'原始图像', u'聚类图像 K=4', u'聚类图像 K=8']
images = [img1, dst41, dst81]
for i in range(3):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i]),
    plt.title(titles[i])
plt.show()