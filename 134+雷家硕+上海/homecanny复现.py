import cv2

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('canny', cv2.Canny(gray, 400, 400))
cv2.waitKey()