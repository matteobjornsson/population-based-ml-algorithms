#################################
# Scratch file for plotting data
#################################

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 


fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')

a, b, c, d, e, f, g, h, i, j = np.loadtxt('./PSO_tune/PSO_tune_abalone.csv', skiprows=1, delimiter=',', unpack=True)
x = np.log(e)
y = np.log(f)
z = j
col = np.arange(len(x))
ax1.set_xlabel('c1')
ax1.set_ylabel('c2')
ax1.scatter(x,y,z, c=col)
ax1.set_zlim3d(0, 0.01)
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