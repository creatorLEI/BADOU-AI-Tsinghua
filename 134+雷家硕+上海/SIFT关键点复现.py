import cv2
import numpy as np

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT()
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51,163,245))

cv2.imshow('sift', img)
cv2.waitKey()