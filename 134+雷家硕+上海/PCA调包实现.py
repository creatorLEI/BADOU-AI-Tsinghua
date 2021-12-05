# import numpy as np
# from sklearn.decomposition import PCA
# x = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
# pca = PCA(n_components=2)#指定函数，降到二维
# pca.fit(x)#训练
# new = pca.fit_transform(x)#由x训练，再由x得到结果
# print(pca.explained_variance_ratio_)    #输出从四维降到二维的贡献率
# print(new)

import numpy as np
from sklearn.decomposition import PCA
x = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
pca = PCA(n_components=2)
pca.fit(x)
new = pca.fit_transform(x)
print(pca.explained_variance_ratio_)
print(new)