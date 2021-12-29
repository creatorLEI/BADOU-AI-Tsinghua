import cv2
import numpy as np

def drawmatchesknncv2(img1_gray, img2_gray, kp1, kp2, goodmatch):
    #图像拼接
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]#读取图像尺度

    vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1+w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodmatch]
    p2 = [kpp.trainIdx for kpp in goodmatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])+(w1,0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))

    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    cv2.imshow('match', vis)

img1_gray = cv2.imread('iphone1.png')
img2_gray = cv2.imread('iphone2.png')
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1_gray, None)#获取关键点以及它的描述
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)#用关键点的描述进行匹配
goodmatch = []#获取好的匹配点
for m,n in matches:
    if m.distance < 0.5*n.distance:
        goodmatch.append(m)

drawmatchesknncv2(img1_gray, img2_gray, kp1, kp2, goodmatch[:20])#绘制关键点
cv2.waitKey()