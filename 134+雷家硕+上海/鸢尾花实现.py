import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets._base import load_iris
#注意这里base前要加_
import matplotlib.pyplot as plt

#加载鸢尾花数据
x, y = load_iris(return_X_y=True)#y是类别，分为0，1，2
pca = PCA(n_components=2)
new = pca.fit_transform(x)#只对x降维了

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(new)):#数据个数
    if y[i] == 0:
        red_x.append(new[i][0])
        red_y.append(new[i][1])
    elif y[i] == 1:
        blue_x.append(new[i][0])
        blue_y.append(new[i][1])
    else:
        green_x.append(new[i][0])
        green_y.append(new[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

