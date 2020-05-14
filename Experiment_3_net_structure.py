import pandas as pd
import tc_net
import os
import priorization
import result_analysis

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

# 读取数据
tc_Stage = pd.read_csv(file_name, sep=';', usecols=["Id", "Stage"], index_col="Id")

times = 3

run_time = []
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

    save_name = str(N_layers) + 'layers_exp3' + '_' + str(times) + '.csv'

    time_cost = priorization.TcData2Result(tc_Stage=tc_Stage, rank_net=rank_net,
                                           PATH_ori=params_path + 'exp3_' + str(N_layers) + '_layers_ori.pth',
                                           PATH=params_path + 'exp3_' + str(N_layers) + '_layers.pth',
                                           N_input=N_input, N_hidden=N_hidden, N_layers=N_layers, history_length=4,
                                           memory_setting=memory_setting, train_setting=train_setting, epoch=1,
                                           learn_rate=learn_rate, alpha=alpha, gamma=gamma,
                                           train_flag=True, param_read=False,
                                           save_model_ori=True, save_model=True,
                                           dataset_name=dataset_name,
                                           data_save_path=data_save_path + '/' + save_name)

    result_analysis.result_analysis(save_path=result_analysis_path,
                                    file_name=data_save_path + '/' + save_name, save_name=save_name,
                                    N_hidden=N_hidden, N_layers=N_layers,
                                    train_setting=train_setting, memory_setting=memory_setting,
                                    dataset_name=dataset_name, save_flag=True, show_flag=False, mini_batch=10)
    run_time.append([str(N_layers)+'layers_'+str(times), time_cost])
    print(str(N_layers) + 'layers训练完成')

run_time = pd.DataFrame(run_time)
run_time.to_csv('./result_data/result_of_exp3' + '/time_cost.csv', mode='a')
