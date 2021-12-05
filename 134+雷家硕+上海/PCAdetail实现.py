# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np


class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):  # self表示都是实例变量，并且是全局变量，即不会封闭在__init__函数中其他地方不能调用
        '''
        :param X,训练样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值

        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        '''为什么上面还要建立空矩阵，直接用底下的不可以吗？'''

        self.centrX = self._centralized()  # 求中心化后的样本
        self.C = self._cov()  # 求中心化后样本的协方差矩阵
        self.U = self._U()  # 基于协方差矩阵求样本的特征向量，并根据目标维度组成新的基底
        self.Z = self._Z()  # 基于新的基底，让原样本重新投影，降维

    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        # 样本集的特征均值
        #指定按特征求均值,因为for一个矩阵，就是按行读取的，所以先转置
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        #注意是利用样本求特征的协方差，所以说转置的那个矩阵应该在前
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(
            self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b[0])
        ind = np.argsort(-1 * a)
        # 给出特征值降序的topK的索引序列，返回的是list
        # 只不过原始特征向量矩阵恰好按降序排列的，所以就是[0，1，2]
        # 也有可能是[1,0,3]，就绝对不是重新index了，就是以前的列数
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        # 虽然你得到的特征向量是列向量，但存的时候是不会按列存的，
        # 都是变成行向量存，所以最终得到的结果是基底的转置
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)
