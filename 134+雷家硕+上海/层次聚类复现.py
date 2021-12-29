from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import matplotlib.pyplot as plt
x = [[1,2],[3,2],[4,4],[1,2],[1,3]]
#计算点之间的距离
z = linkage(x,'ward')
#开始聚类
f = fcluster(z, 4, 'distance')#第二个参数为聚类数的阈值，最多聚成4类，毕竟一共才5个点
dn = dendrogram(z)
# plt.figure(figsize=(5,3))
plt.show()
