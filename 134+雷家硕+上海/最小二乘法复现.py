import numpy as np
import pandas as pd

data = pd.read_csv('train_data.csv')
x = data['X'].values
y = data['Y'].values

s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = len(x)

for i in range(n):
    s1 = s1 + x[i]*y[i]
    s2 = s2 + x[i]
    s3 = s3 + y[i]
    s4 = s4 + x[i]**2

k = (n*s1 - s2*s3)/(n*s4 - s2**2)
b = (s3 - k*s2)/n

print('k:{} b:{}'.format(k,b))#字符串格式化方法，可以用来替代%foramt会把参数按位置顺序来填充到字符串中
