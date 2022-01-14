import cv2
import numpy as np


def get_ahash(img):
    img_reset = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_reset,cv2.COLOR_BGR2GRAY)

    h,w = gray.shape
    sum_ = 0
    for i in range(h):
        for j in range(w):
            sum_ += gray[i][j]
    mean_ = sum_/(h*w)

    hash_ = []
    for k in range(h):
        for g in range(w):
            if gray[k][g] < mean_:
                hash_.append(0)
            else:
                hash_.append(1)
    hash_arr = np.array(hash_)
    return hash_arr

def get_dhash(img):
    img_reset = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)#8*(要写成(9,8)
    gray = cv2.cvtColor(img_reset,cv2.COLOR_BGR2GRAY)
    hash_ = []
    for i in range(8):
        for j in range(8):
            if gray[i,j+1] - gray[i,j] < 0:
                #像素值大于后一个像素
                hash_.append(1)
            else:
                hash_.append(0)
    hash_arr = np.array(hash_)
    return hash_arr

def get_hanming(hash_arr1,hash_arr2):
    dis = (hash_arr2 - hash_arr1)**2
    num = sum(dis)
    return dis,num


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
# ahash_arr1 = get_ahash(img1)
# ahash_arr2 = get_ahash(img2)
# adis,anum = get_hanming(ahash_arr1,ahash_arr2)
# print(adis,anum)
dhash_arr1 = get_dhash(img1)
dhash_arr2 = get_dhash(img2)
ddis,dnum = get_hanming(dhash_arr1,dhash_arr2)
print(dnum)
