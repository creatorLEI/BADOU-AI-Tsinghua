import numpy as np
import random
from collections import defaultdict
x = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]

length = len(x)
cluster_num = 3
def init_centroid(cluster_num,length,x):
    idx_list = random.sample(range(length),cluster_num)#直接就是list
    init_centroid_list = []
    for i in idx_list:
        init_centroid_list.append(x[i])
    return init_centroid_list


def init_cluster(init_centroid_list,y):#y是筛选好的，没有质心的x
    centroid_array = np.array(init_centroid_list)
    init_cluster_dict = defaultdict(list)
    for i in y:
        point_array = np.tile(np.array(i),(cluster_num,1))
        tmp_array = np.sum((point_array-centroid_array)**2,axis=1)
        dis_array = tmp_array**(1/2)
        for idx, j in enumerate(dis_array):
            if j == min(dis_array):
                init_cluster_dict[idx].append(i)
                break
    return init_cluster_dict

def new_centroid(init_cluster_dict):
    new_centroid_array = np.zeros([3,2])
    for key, points in init_cluster_dict.items():
        points_array = np.array(points)
        centroid_new = np.mean(points_array,axis=0)
        new_centroid_array[key] = centroid_new
    new_centroid_list = list(new_centroid_array)
    return new_centroid_list



def iteration_cluster(init_cluster_dict,new_centroid_list,max_iteration,x):
    new_cluster_dict = init_cluster(new_centroid_list, x)
    for i in range(max_iteration):
        if (init_cluster_dict == new_cluster_dict) or (i == max_iteration-1):
            return new_cluster_dict
        else:
            new_centroid_list = new_centroid(new_cluster_dict)
            init_cluster_dict = new_cluster_dict#将新分类作为对比分类
            new_cluster_dict = init_cluster(new_centroid_list,x)

init_centroid_list = init_centroid(cluster_num,length,x)
y = []
for i in x:
    if i not in init_centroid_list:
        y.append(i)
# print(y)
init_cluster_dict = init_cluster(init_centroid_list,y)
new_centroid_list = new_centroid(init_cluster_dict)
final_cluster_dict = iteration_cluster(init_cluster_dict, new_centroid_list, 200, x)
print(final_cluster_dict)















