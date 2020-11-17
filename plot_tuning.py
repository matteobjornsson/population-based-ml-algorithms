#################################
# Scratch file for plotting data
#################################

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')

x, y, z = np.loadtxt('./tuning/kmeans_segmentation.csv', skiprows=1, delimiter=',', unpack=True)
col = np.arange(len(x))
ax1.set_xlabel('Neighbors')
ax1.set_ylabel('Bin Width')
ax1.scatter(x,y,z, c=col)
plt.show()

# x, y, z = np.loadtxt('./knn_tuning/knn_tuning_abalone(k-bin).csv',skiprows=1, delimiter=',', unpack=True)
# x = np.log(x)
# y = np.log(y)
# ax1.set_xlabel('k_value')
# ax1.set_ylabel('bin_with')
# col = np.arange(len(x))
# ax1.scatter(x,y,z, c=col)
# # plt.xscale('log')
# plt.show()

# x, y, z = np.loadtxt('./knn_tuning/knn_tuning_abalone(k-ratio).csv',skiprows=1, delimiter=',', unpack=True)
# x = np.log(x)
# y = np.log(y)
# ax1.set_xlabel('k_value')
# ax1.set_ylabel('ratio')
# ax1.scatter(x,y,z)
# plt.show()