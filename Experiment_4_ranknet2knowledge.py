import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

import tc_net


def Observation_3D(RunTime, FreeTime, LastResults):
    # 观察智能体的排序函数的3D曲面图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=45, azim=225)

    X, Y = np.meshgrid(RunTime, FreeTime)
    Z = np.zeros([len(RunTime), len(FreeTime)])
    for k in range(1):
        print('开始计算Z~')
        with torch.no_grad():
            for i in range(len(RunTime)):
                for j in range(len(FreeTime)):
                    v_input = [RunTime[i], FreeTime[j]] + LastResults[k]
                    v_input = torch.tensor(v_input)

                    v_output = rank_net(v_input)
                    Z[i, j] = v_output.detach().numpy()
        print('计算结束，准备画图')
        p = ax.plot_surface(X, Y, Z)
        # ax.contour(X, Y, Z, zdir = 'z', offset = 0, cmap = plt.get_cmap('rainbow'))
        # fig.colorbar(p)

    ax.set_xlabel('Duartion')
    ax.set_ylabel('FreeTime')
    ax.set_zlabel('Rank value')
    ax.set_title('')
    plt.show()


def ViewTrend(V1, V2, V3, main_col='RunTime',view=True,result_save_path=None):
    Z = np.zeros(len(V2))
    v1 = V1[0]
    plt.figure(1)
    for k in range(len(V3)):
        for j in range(len(V2)):
            if main_col == 'FreeTime':
                v_input = [v1, V2[j]] + V3[k]
            elif main_col == 'RunTime':
                v_input = [V2[j], v1] + V3[k]
            else:
                break
            v_input = torch.tensor(v_input)

            v_output = rank_net(v_input)
            Z[j] = v_output.detach().numpy()

        plt.plot(V2, Z, label=''.join(list(map(str, V3[k]))))

    plt.xlabel(main_col)
    plt.ylabel('Rank value')
    plt.legend()
    if result_save_path is not None:
        plt.savefig(result_save_path+main_col+'.jpg')
    if view:
        plt.show()
    plt.cla()


def ViewCompare(RunTime,FreeTime,V3,view=True,result_save_path=None):
    main_col_list = ['RunTime', 'FreeTime']
    plt.figure(1)
    for main_col in main_col_list:
        k = 0
        if main_col == 'FreeTime':
            V1 = RunTime
            V2 = FreeTime
            Z = np.zeros(len(V2))
        elif main_col == 'RunTime':
            V1 = FreeTime
            V2 = RunTime
            Z = np.zeros(len(V2))
        else:
            break

        v1 = V1[0]
        for j in range(len(V2)):
            if main_col == 'FreeTime':
                v_input = [v1, V2[j]] + V3[k]
            elif main_col == 'RunTime':
                v_input = [V2[j], v1] + V3[k]
            else:
                break
            v_input = torch.tensor(v_input)

            v_output = rank_net(v_input)
            Z[j] = v_output.detach().numpy()

        plt.plot(V2, Z, label=main_col)

    plt.xlabel('x')
    plt.ylabel('Rank value')
    plt.legend()
    if result_save_path is not None:
        plt.savefig(result_save_path + 'compare.jpg')
    if view:
        plt.show()
    plt.cla()


param_read = './result_data/result_of_exp2/params/exp2_0.5_0.8_0.01.pth'
result_save_path = 'result_data/result_of_exp4'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
result_save_path = result_save_path+'/'
rank_net = tc_net.Rank_net(6, 1, 24)
rank_net.load_state_dict(torch.load(param_read))

RunTime = np.arange(0, 1, 0.01)
FreeTime = np.arange(0, 1, 0.01)
LastResults = []
for i in range(16):
    LastResults.append(list(map(int, list('{:04b}'.format(i)))))

V1 = RunTime
V2 = FreeTime
V3 = LastResults
main_col = 'FreeTime'
ViewTrend(V1,V2,V3,main_col=main_col,result_save_path=result_save_path)

V1 = FreeTime
V2 = RunTime
V3 = LastResults
main_col = 'RunTime'
ViewTrend(V1,V2,V3,main_col=main_col,result_save_path=result_save_path)

ViewCompare(RunTime,FreeTime,LastResults,result_save_path=result_save_path)

v1 = V1[-1]
v2 = V2[-1]
Z = np.zeros(len(LastResults))
for k in range(len(LastResults)):
    v_input = [v1,v2] + LastResults[k]
    v_input = torch.tensor(v_input)

    v_ouput = rank_net(v_input)
    Z[k] = v_ouput.detach().numpy()
plt.figure(1)
plt.bar([''.join(map(str,LastResults[i])) for i in range(len(LastResults))],Z)
plt.xlabel('LastResults')
plt.ylabel('Rank value')
plt.show()
plt.cla()
