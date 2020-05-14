import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import os

# 数据
data = pd.read_excel('./result_data/result_of_exp2/result_analysis/std_exp2.xlsx')
data_save_path = './result_data/result_of_exp2/result_des'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

x = data.iloc[:, 0]
y = data.iloc[:, 1]
z = data.iloc[:, 2]
z_mean = data.iloc[:, 3]
z_std = data.iloc[:, 4]
z_residual = data.iloc[:, 5]
z_c = [z_mean, z_residual, z_std]
legend = ['NAPFD_mean','NAPFD_std','NAPFD_residual']
# 绘制散点图
for i in range(3):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=100, c=z_c[i], cmap=plt.cm.rainbow, alpha=1)
    ax.set_zlabel('$\eta$')
    ax.set_ylabel('$\gamma$')
    ax.set_xlabel('$\\alpha$')
    fig.savefig(data_save_path + '/' + legend[i] + '.jpg')
    plt.cla()

