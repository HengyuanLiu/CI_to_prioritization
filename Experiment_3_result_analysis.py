import pandas as pd
import tc_net
import os
import priorization
import result_analysis
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset_name = "paintcontrol"
file_name = r"./data/" + dataset_name + ".csv"
data_save_path = './result_data/result_of_exp3/'
params_path = data_save_path + 'params/'
result_analysis_path = data_save_path + 'result_analysis'

N_input = 6
N_hidden = 24
N_output = 1
history_length = 4

learn_rate = 0.01
alpha = 0.5
gamma = 0.5

train_setting = 'DQN'
memory_setting = 'current'

N_layers_list = [3, 6, 9]

# 同实验三数据的times
times = 3
std_list = []


for N_layers in N_layers_list:
    if N_layers == 3:
        rank_net = tc_net.Rank_net(N_input, N_output, N_hidden)
    elif N_layers == 6:
        rank_net = tc_net.Rank_net_SixLayers(N_input, N_output, N_hidden)
    elif N_layers == 9:
        rank_net = tc_net.Rank_net_NineLayers(N_input, N_output, N_hidden)
    else:
        print('非实验预定排序网络层数')
        break

    NAPFD = pd.DataFrame()
    index_list = []
    NAPFD_col = []
    for time in range(1,times+1):
        save_name = str(N_layers) + 'layers_exp3' + '_' + str(time) + '.csv'
        data_path = data_save_path + save_name
        data = pd.read_csv(data_path, sep=';', index_col='Id')

        data_class = result_analysis.data_set(data)
        if time == 1:
            index_list = list((data_class.NAPFD_sup().NAPFD > 1e-6) & (data_class.NAPFD_inf().NAPFD < 1 - 1e-6))
            NAPFD = data_class.NAPFD_rank()
            NAPFD = NAPFD.rename(columns={'NAPFD': 'NAPFD' + '_' + str(time)})
            NAPFD_max = data_class.NAPFD_sup()
            NAPFD_min = data_class.NAPFD_inf()
        else:
            temp = data_class.NAPFD_rank()
            temp = temp.rename(columns={'NAPFD': 'NAPFD' + '_' + str(time)})
            NAPFD = pd.merge(NAPFD, temp, on='Stage')

        NAPFD_col.append('NAPFD' + '_' + str(time))

    NAPFD['NAPFD_mean'] = NAPFD.loc[:, NAPFD_col].mean(1)
    # NAPFD数据取需要排序的阶段
    NAPFD = NAPFD.loc[index_list, :]
    NAPFD_max = NAPFD_max.loc[index_list, :]
    NAPFD_min = NAPFD_min.loc[index_list, :]

    x = NAPFD.Stage.values.reshape(-1, 1)
    y = NAPFD.NAPFD_mean.values.reshape(-1, 1)
    y_max = NAPFD_max.NAPFD.values.reshape(-1, 1)
    y_min = NAPFD_min.NAPFD.values.reshape(-1, 1)
    model = LinearRegression()
    model_max = LinearRegression()
    model_min = LinearRegression()
    model.fit(x, y)
    model_max.fit(x, y_max)
    model_min.fit(x, y_min)
    y_pred = model.predict(x)
    y_max_pred = model_max.predict(x)
    y_min_pred = model_min.predict(x)
    plt.figure(1)
    plt.plot(x, y, 'b-')
    plt.plot(x, y_pred, 'b-', label='NAPFD_mean')
    plt.plot(x, y_max, 'r--')
    plt.plot(x, y_max_pred, 'r-', label='NAPFD_max')
    plt.plot(x, y_min, 'g--')
    plt.plot(x, y_min_pred, 'g-', label='NAPFD_min')
    plt.legend()
    plt.xlabel('Stage')
    plt.ylabel('NAPFD')
    plt.title(str(N_layers) + 'layers')
    plt.savefig(result_analysis_path + '/' + str(N_layers) + 'layers_exp3.jpg')
    plt.cla()

    plt.figure(2)
    if N_layers == 3:
        plt.plot(x,y_max_pred,label='NAPFD_max')
        plt.plot(x,y_min_pred,label='NAPFD_min')
    plt.plot(x, y_pred, label = str(N_layers) + 'layers')


    std_list.append([str(N_layers) + 'layers', NAPFD.NAPFD_mean.mean(), NAPFD.NAPFD_mean.std()])

plt.xlabel('Stage')
plt.ylabel('NAPFD')
plt.legend()
plt.savefig(result_analysis_path + '/' + 'all_in.jpg')
plt.cla()

std_list = pd.DataFrame(std_list)
std_list.to_excel(result_analysis_path + '/std_exp3.xlsx')