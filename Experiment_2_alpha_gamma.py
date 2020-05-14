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

learn_rate_list = [0.001, 0.01, 0.1]
alpha = 0.5  # [0.2,0.5,0.8]
# gamma = 0.2  # [0.2,0.5,0.8]
gamma_list = [0.2, 0.5, 0.8]

train_setting = "DQN"
memory_setting = "current"

data_save_path = './result_data/result_of_exp2'
result_analysis_path = data_save_path + '/result_analysis'
params_path = data_save_path + '/params'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

run_time = []

for learn_rate in learn_rate_list:
    for gamma in gamma_list:
        save_name = str(alpha) + '_' + str(gamma) + '_' + str(learn_rate) + '_exp2' + '.csv'
        # 重新定义持续集成阶段
        tc_Stage = pd.read_csv(file_name, sep=';', usecols=["Id", "Stage"], index_col="Id")

        time_cost = priorization.TcData2Result(
            tc_Stage=tc_Stage,
            PATH=params_path + '/' + 'exp2_' + str(alpha) + '_' + str(gamma) + '_' + str(learn_rate) + '.pth',
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
        run_time.append([str(alpha) + '_' + str(gamma) + '_' + str(learn_rate), time_cost])
        print('exp2_' + str(alpha) + '_' + str(gamma) + '_' + str(learn_rate), '训练完成')

        # 记录已经使用过的参数
        with open("./result_data/result_of_exp2/params_done.txt", "a") as file:
            params_done = 'alpha=' + str(alpha) + ',gamma=' + str(gamma) + ',lr=' + str(learn_rate) + '\n'
            file.write(params_done)

run_time = pd.DataFrame(run_time)
run_time.to_csv('./result_data/result_of_exp2' + '/time_cost.csv', mode='a')

