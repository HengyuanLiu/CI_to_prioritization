import os
import pandas as pd
import priorization
import result_analysis

dataset_name = "paintcontrol"
file_name = r"./data/" + dataset_name + ".csv"
N_input = 6
N_hidden = 24
N_output = 1
N_layers = 3
history_length = 4

learn_rate = 0.01
alpha = 0.5
gamma = 0.5

train_list = ["DNN", "DQN"]
memory_list = ["current", "replay"]
data_save_path = './result_data/result_of_exp1'
params_path = data_save_path + '/params'
result_analysis_path = data_save_path + '/result_analysis'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

run_time = []
for train_setting in train_list:
    for memory_setting in memory_list:
        save_name = train_setting + '_' + memory_setting + '_' + dataset_name + '_exp1' + '.csv'
        # 重新定义持续集成阶段
        tc_Stage = pd.read_csv(file_name, sep=';', usecols=["Id", "Stage"], index_col="Id")

        time_cost = priorization.TcData2Result(tc_Stage=tc_Stage,
                                               PATH=params_path+'/'+'exp1_'+train_setting+'_'+memory_setting+'.pth',
                                               N_input=N_input, N_hidden=N_hidden, N_layers=N_layers, history_length=4,
                                               memory_setting=memory_setting, train_setting=train_setting, epoch=1,
                                               learn_rate=learn_rate, alpha=alpha, gamma=gamma,
                                               train_flag=True, param_read=True, save_model=True,
                                               dataset_name=dataset_name,
                                               data_save_path=data_save_path + '/' + save_name)
        # 数据结果默认文件名格式 网络结构_网络层数_隐藏节点个数_数据集_网络学习率_alpha_gamma_epoch_reward-times

        result_analysis.result_analysis(save_path=result_analysis_path,
                                        file_name=data_save_path + '/' + save_name, save_name=save_name,
                                        N_hidden=N_hidden, N_layers=N_layers,
                                        train_setting=train_setting, memory_setting=memory_setting,
                                        dataset_name=dataset_name, save_flag=True, show_flag=False, mini_batch=10)
        run_time.append([train_setting+'_'+memory_setting, time_cost])
        print('train_setting:' + train_setting + ',memory_setting:' + memory_setting,'训练完成')

run_time = pd.DataFrame(run_time)
run_time.to_csv('./result_data/result_of_exp1' + '/time_cost.csv', mode='a')
