import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
canny = cv2.Canny(gray, 200, 300)
plt.subplot(2,3,1), plt.imshow(gray, 'gray'), plt.title('original')
plt.subplot(2,3,2), plt.imshow(sobel_x, 'gray'), plt.title('sobel_x')
plt.subplot(2,3,3), plt.imshow(sobel_y, 'gray'), plt.title('sobel_y')
plt.subplot(2,3,4), plt.imshow(laplace, 'gray'), plt.title('laplace')
plt.subplot(2,3,5), plt.imshow(canny, 'gray'), plt.title('canny')
plt.show()