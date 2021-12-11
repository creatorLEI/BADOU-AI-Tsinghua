import numpy as np


def WarpPerspectiveMatrix(src, dst):
    #src 输入点阵
    #dst 输出点阵
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4   #只有符合这个声明程序才会运行
            #注意是声明维度，所以一定要加shape
    nums = src.shape[0] #读取点数
    A = np.zeros((2 * nums, 8))  # A是输入x,y组成的矩阵
                                # A*warpMatrix=B；每个点有x,y两维，所以每个点都有2个方程，4个点就是8个方程，5个点就是10个
                                #第二维就是要求的8个未知参数，虽然有9个参数，但是a33已经设为1了
    B = np.zeros((2 * nums, 1)) #B是输出组成的矩阵
                                #输出矩阵

    #A,B都不是要求的warpmatirx!!!

    for i in range(0, nums):    #对于每个点
        A_i = src[i, :] #读取一个输入点，包括x,y
        B_i = dst[i, :] #读取一个输出点
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    #matrix 和 ndarray 都可以通过objects后面加.T 得到其转置。但是 matrix objects 还可以在后面加 .H f得到共轭矩阵, 加 .I 得到逆矩阵。
    #ndarray可以是多维向量，但是matirx一定是2维
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
                            #这是矩阵

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
                                #取[0]的话就是只要1行的数，否则只有T还是一个1*8的向量
                                #先转化为向量，不能matirx直接转置，matrix[0]不能降维为数组，因为matrix永远都是二维矩阵
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
                                        #不能直接是shape，虽然就是8个数，但shape的格式是(8，空值)，所以是要取第一个元素
                                                            #直接1.0就能声明是浮点数了
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src) #list没有shpae，必须先转成shape

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
