import pandas as pd
import numpy as np
import scipy as sp
import scipy.linalg as sl
'''
整个RANSAC function中应该包括
1.随机选取点的函数
2.回归
3.计算残差并记录内群点个数的函数
'''

def random_choice(n,data):
    '''
    随机选择点函数
    :param n:要随机选取的点的个数
    :param data: 待选数据
    :return: 只是返回打乱后的索引值，没有选择数据
    '''
    all_idx = np.arange(data)#获取数据的index，可以不要求原始数据为dataframe，数组也可以
    np.random.shuffle(all_idx)#打乱数据下标
    idx1 = all_idx[:n]
    idx2 = all_idx[n:]
    return idx1,idx2

def leastsquares(x, y):
    '''
    :param x: 随机指定的内群点的所有x坐标
    :param y: 随机指定的内群点的所有y坐标
    :return: 如果x,y传入的是矩阵，那么就可以得到y_pred矩阵
    '''
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    n = len(x)

    for i in range(n):
        s1 = s1 + x[i] * y[i]
        s2 = s2 + x[i]
        s3 = s3 + y[i]
        s4 = s4 + x[i] ** 2

    k = (n * s1 - s2 * s3) / (n * s4 - s2 ** 2)
    b = (s3 - k * s2) / n
    return k,b

def collect_inner(k,b,else_x,else_y,r):
    '''
    :param k: 当前预测模型的参数
    :param b: 当前预测模型的参数
    :param else_x: 原始数据中除去随机选中用来拟合的点,list
    :param else_y: 原始数据中除去随机选中用来拟合的点,list
    :param r:用来选取内群点的可接受误差
    :return: 该函数额外的内群点的数量
    '''
    inner_num = 0
    for x_i, y_i in zip(else_x,else_y):
        y_pred = k * x_i + b
        error_i = (y_i-y_pred)**2
        if error_i <= r:
            inner_num += 1
    return inner_num

def ransac(itera,n,r,data):
    '''
    迭代的过程用ransac实现
    要实现：
        规定迭代次数itera
        指定随机选取的点的个数n
        从原始数据中随机选取n个x,y
        用选出来的x,y leastsquares线性方程
        计算该线性方程的内群点个数collect_inner
        记录该线性方程和内群点个数，与下一个方程的其内群点个数比较，如果小于，暂用下一个方程
    :return: 最合适的方程的k,b
    '''
    #从原始数据中随机选取点
    inner_num_pro = 0
    coff = None
    best_dict = {}
    best_dict['bestcoff'] = coff
    for i in range(itera):
        idx1,idx2 = random_choice(n,data.shape[0])#idx1是用来拟合方程的，idx2是用来计算内群点的
        x = data[idx1,0]
        y = data[idx1,1]
        else_x = data[idx2,0]
        else_y = data[idx2,1]
        #用选出来的点拟合
        k,b = leastsquares(x, y)
        #计算该线性方程的内群点个数collect_inner
        inner_num = collect_inner(k, b, else_x, else_y, r)
        if inner_num > inner_num_pro:
            coff = (k,b)
            best_dict['bestcoff'] = coff
        inner_num_pro = inner_num
    return best_dict

#构建数据
def data_build():
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        #有时候可能想测试一段代码，加个为真的条件语句方便调试，不用了直接为0就行
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    return data

data = data_build()
best = ransac(1000,50,7e3,data)
print(best)









