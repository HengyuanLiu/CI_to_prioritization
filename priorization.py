import pandas as pd
import tc_data_reconstitution as tcDataRecon
import tc_net as tcNet

import torch
import torch.nn as nn
import torch.optim as optim
import reward_func as reward_F
import result_analysis
import copy
import time
import random


def TcData_std(tc_data_col, std_setting='max-min'):
    """输入给定数据列"""
    # max-min标准化
    if std_setting == 'max-min':
        if tc_data_col.max() - tc_data_col.min() < 1e-6:
            if tc_data_col.max() < 1e-6:
                tc_data_col = 0
            else:
                tc_data_col = 1
        else:
            tc_data_col = (tc_data_col - tc_data_col.min())/(tc_data_col.max() - tc_data_col.min())
    return tc_data_col


def TcData_filled(tc_data_col):
    """将输入数据中的零值进行随机填充"""
    tc_data_col = tc_data_col.astype(dtype=float)
    for index in list(tc_data_col.index):
        if tc_data_col[index] == 0 or tc_data_col[index] < 1e-6:
            tc_data_col[index] = random.random()
    return tc_data_col


def TcData2Result(tc_Stage=None, rank_net=None, PATH=None, PATH_ori=None,
                  N_input=None, N_output=None, N_hidden=None, N_layers=None, dataset_name=None,
                  data_save_path=None, save_name='tc_result.csv', memory_setting=None,
                  train_setting=None, train_flag=None, param_read=None, save_model_ori=None, save_model=None,
                  history_length=None, alpha=0.5, gamma=0.5, learn_rate=0.2, epoch=1, NAPFD_view=False):
    """
    根据可选参数实现深度强化学习的测试用例排序
    :param PATH: 模型训练结果参数保存路径
    :param PATH_ori: 模型初始参数保存路径
    :param rank_net: 模型输入
    :param epoch: 记忆回放次数
    :param learn_rate: 排序网络学习率
    :param alpha: 奖励作用程度
    :param gamma: 预期奖励折扣
    :param tc_Stage: 数据的阶段标签
    :param N_input: 智能体决策网络输入向量维度
    :param N_output: 智能体决策网络输出向量维度
    :param N_hidden: 智能体决策网络隐藏层节点个数
    :param N_layers: 智能体决策网络层数
    :param dataset_name: 数据集名称
    :param data_save_path: 排序结果存储路径
    :param save_name: 结果保存
    :param stage_flag: 根据阶段的区分标签生成阶段标签
    :param memory_setting: 记忆存储类型
    :param train_setting: 智能体的学习类型
    :param train_flag: 是否进行训练
    :param param_read: 参数读取
    :param save_model_ori: 是否保存网络初始参数
    :param save_model: 是否保存网络的最终参数
    :param history_length: 使用的历史信息长度
    :return: 生成测试用例排序结果文件
    """
    if dataset_name is None:
        print("没有给定数据集,将采用默认数据集iofrol.csv")
        dataset_name = "iofrol"
    file_name = r"./data/" + dataset_name + ".csv"

    if data_save_path is None:
        data_save_path = './result/result_data/result_of_' + dataset_name + '/' + save_name
    # 网络参数设定
    if history_length is None:
        history_length = 4  # 使用历史信息长度

    if N_input is None:
        N_input = 6  # 网络输入向量维度
    if N_output is None:
        N_output = 1  # 网络输出向量维度
    if N_hidden is None:
        N_hidden = 12
    if N_layers is None:
        N_layers = 2

    if train_flag is None:
        train_flag = True  # 是否训练
    if param_read is None:
        param_read = False  # 是否使用现有参数
    if save_model_ori is None:
        save_model_ori = False  # 是否保存网络初始权重
    if save_model is None:
        save_model = False  # 是否保存网络权重
    if memory_setting is None:
        memory_setting = "current"
    if train_setting is None:
        train_setting = "DNN"
    if PATH is None:
        PATH = './params/' + train_setting + '_' + memory_setting + '_' + str(N_layers) + '_' + \
               str(N_hidden) + '_' + dataset_name + '_' + str(learn_rate) + '_' + str(alpha) + \
               '_' + str(gamma) + '_' + str(epoch) + '.pth'  # 模型保存路径
    # PATH_ori = './params/' + train_setting + '_' + memory_setting + '_' + str(N_layers) + '_' + \
    #            str(N_hidden) + '_' + dataset_name + '_ori.pth'  # 模型保存路径
    if PATH_ori is None:
        PATH_ori = './params/' + dataset_name + '.pth'  # 模型保存路径
    # 排序网络所使用的数据
    data_list = ['Duration', 'RunTime', 'FreeTime'] + \
                [('LastResult' + str(R_name)) for R_name in range(1, history_length + 1)]
    data_list_act = ['RunTime', 'FreeTime'] + \
                    [('LastResult' + str(R_name)) for R_name in range(1, history_length + 1)]
    data_list_learn = ['Duration', 'FreeTime', 'Verdict'] + \
                      [('LastResult' + str(R_name)) for R_name in range(1, history_length)]

    NAPFD_frame = pd.DataFrame(columns=['Stage', 'NAPFD'])

    if memory_setting == "current":
        TS_minimal = 5  # 只有当次经验记忆时,当本次测试套件规模大于TS_minimal时智能体才进行学习
    elif memory_setting == "replay":
        memory_capacity = 10000  # 具有记忆能力时的记忆容量
        memory_replay = 1000  # 每次回放次数
    else:
        pass
    memory_storage = pd.DataFrame(columns=data_list)

    # 重新定义持续集成阶段
    if tc_Stage is None:
        tc_Stage = pd.read_csv(file_name, sep=';', usecols=["Id", "Stage"], index_col="Id")

    tc_Name = pd.read_csv(file_name, sep=';', usecols=["Id", "Name"], index_col="Id")

    # 定义排序网络
    if rank_net is None:
        rank_net = tcNet.Rank_net(N_input, N_output, N_hidden)

    # 是否使用已经训练好的模型
    if not param_read:
        # 网络参数初始化
        tcNet.initNetParams(rank_net)        # 读取训练好的模型
    else:
        rank_net.load_state_dict(torch.load(PATH_ori))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rank_net.parameters(), lr=learn_rate)

    # 保存初始参数
    if save_model_ori:
        print("网络初始参数为:\n", list(rank_net.parameters()))
        torch.save(rank_net.state_dict(), PATH_ori)

    if train_setting == "DQN":
        # 构建目标网络
        rank_net_target = copy.deepcopy(rank_net)

    print("此数据包含", str(tc_Stage.Stage.max()), "个阶段")
    time_start = time.clock()
    print("程序起始运行时间为", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for stage in range(1, max(tc_Stage.Stage) + 1):
        """读入本阶段数据"""
        # 读取指定阶段的测试数据
        tc_data_stage = tcDataRecon.tc_data_stage_read(tc_Stage, stage, file_name)
        tc_data_stage['Stage'] = stage
        start_index = tc_data_stage.index.min()

        # 产生网络输入数据
        run_time_average = 0
        run_time_std = 0
        if not memory_storage.empty:
            run_time_average = memory_storage.Duration.mean()
            run_time_std = memory_storage.Duration.std()

        tc_data_stage = tcDataRecon.tc_data_generalize_FreeTime(tc_data_stage, tc_Name, file_name)
        tc_data_stage = tcDataRecon.tc_data_generalize_RunTime(tc_data_stage, tc_Name, file_name,
                                                               run_time_average, run_time_std)
        tc_data_stage = tcDataRecon.tc_LastResults_split(tc_data_stage, history_length=history_length)

        # 加入结果保存列
        tc_data_stage = tc_data_stage.reindex(columns=list(tc_data_stage.columns) + ['Rank'], fill_value=0)

        # 更新记忆存储库
        if memory_setting == 'current':
            memory_storage = tc_data_stage
        elif memory_setting == 'replay':
            memory_storage = pd.concat([memory_storage, tc_data_stage], ignore_index=True, sort=True)
            # 将记忆存储库进行删减
            index = 0
            while memory_storage.shape[0] > memory_capacity:
                memory_storage = memory_storage.drop(index=index)
                index += 1
            memory_storage.reset_index(drop=True)
        else:
            pass

        # 对更新后的记忆存储库进行max-min标准化
        memory_storage_std = memory_storage.copy(deep=True)
        memory_storage_std.FreeTime = TcData_std(memory_storage_std.FreeTime, std_setting='max-min')

        memory_storage_std.RunTime = TcData_std(memory_storage_std.RunTime, std_setting='max-min')
        memory_storage_std.RunTime = TcData_filled(memory_storage_std.RunTime)

        memory_storage_std.Duration = TcData_std(memory_storage_std.Duration, std_setting='max-min')

        tc_data_stage.FreeTime = TcData_std(tc_data_stage.FreeTime, std_setting='max-min')
        tc_data_stage.RunTime = TcData_std(tc_data_stage.RunTime, std_setting='max-min')
        tc_data_stage.RunTime = TcData_filled(tc_data_stage.RunTime)

        """智能体决策"""
        # 将DataFrame转换为torch.tensor
        data_vec = torch.tensor(
            tc_data_stage.loc[:, data_list_act].apply(pd.to_numeric).values,
            dtype=torch.float, requires_grad=True)  # 网络输入值

        # 对输入的测试用例输出rank
        with torch.no_grad():
            for index in range(len(data_vec)):
                # 每个样本数据的处理
                data_input = data_vec[index].clone().detach()
                data_output = rank_net(data_input)
                tc_data_stage.loc[start_index + index, 'Rank'] = data_output.item()

        if stage == 1:
            tc_data_stage.to_csv(data_save_path,
                                 columns=['Name', 'Duration', 'Verdict', 'Rank', 'Stage', 'Cycle'],
                                 float_format="%.3f", sep=';')
        else:
            tc_data_stage.to_csv(data_save_path, mode='a',
                                 columns=['Name', 'Duration', 'Verdict', 'Rank', 'Stage', 'Cycle'],
                                 header=False,
                                 float_format="%.3f", sep=';')

        # 计算排序效果
        if NAPFD_view and stage % 25 == 0 and stage >= 25:
            NAPFD_frame.loc[stage, 'Stage'] = stage
            NAPFD_frame.loc[stage, 'NAPFD'] = result_analysis.tc_NAPFD(tc_data_stage)
            result_analysis.result_analysis(N_hidden=N_hidden, N_layers=N_layers, save_name=save_name,
                                            train_setting=train_setting, memory_setting=memory_setting,
                                            dataset_name=dataset_name,
                                            save_flag=False, show_flag=True, plot_NAPFD=False, plot_NAPFD_adj=True,
                                            mini_batch=10)

        """智能体学习"""
        if train_flag:
            # 记忆读取
            for epoch_curr in range(epoch):
                if memory_setting == 'current':  # 当前轮次经验学习
                    data_replay = memory_storage_std
                elif memory_setting == 'replay':  # 记忆回放
                    if memory_storage_std.shape[0] >= memory_replay:
                        data_replay = memory_storage_std.sample(memory_replay)
                    else:
                        data_replay = memory_storage_std
                else:
                    pass
                # 智能体开始学习
                for index in list(data_replay.index):
                    rank_net.zero_grad()  # 梯度清零
                    # 将测试用例数据转化为tensor
                    data_input = torch.tensor(
                        data_replay.loc[index, data_list_learn],
                        dtype=torch.float, requires_grad=True
                    )
                    # 计算测试用例的奖励值
                    reward = reward_F.reward_function(data_replay.loc[index, 'Verdict'], function_name='tc_reward')
                    reward = torch.tensor(reward, dtype=torch.float, requires_grad=False)

                    data_output = rank_net(data_input)

                    if train_setting == 'DNN':
                        # DNN:Q = Q + alpha * r
                        data_target = data_output.clone().detach() + alpha * reward
                    elif train_setting == 'DQN':
                        # DQN:Q = Q + alpha * （r + gamma * Q' - Q）
                        data_target = data_output.clone().detach() + \
                                      alpha * (reward + gamma * rank_net_target(data_input) - data_output)
                    else:
                        pass

                    loss = criterion(data_output, data_target)
                    loss.backward()
                    optimizer.step()
                # 智能体学习结束

                # 如果训练类型为DQN那么更新目标网络
                if train_setting == 'DNN':
                    break  # 仅训练一次
                elif train_setting == 'DQN':
                    rank_net_target = copy.deepcopy(rank_net)
                else:
                    pass

            print("第" + str(stage) + "阶段学习完成,awa~")

    # 如果需要保存训练参数,那么保存训练后
    if save_model:
        torch.save(rank_net.state_dict(), PATH)

    time_end = time.clock()
    print("程序终止运行时间为", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序耗时', time_end - time_start)
    return time_end - time_start


if __name__ == '__main__':  # 根据设定的参数具体数据结果
    # 主要程序
    dataset_name = "paintcontrol"
    # dataset_name = "iofrol"
    file_name = r"./data/" + dataset_name + ".csv"

    N_input = 6
    N_hidden = 24
    N_output = 1
    N_layers = 3
    history_length = 4
    train_setting = "DQN"
    memory_setting = "current"
    if train_setting == "DNN" and memory_setting == "current":
        param_read = False
    else:
        param_read = True
    learn_rate = 0.01
    alpha = 0.5
    gamma = 0.5
    save_name = train_setting + '_' + memory_setting + '_' + \
                str(N_layers) + '_' + str(N_hidden) + '_' + dataset_name + '.csv'
    # 重新定义持续集成阶段
    tc_Stage = pd.read_csv(file_name, sep=';', usecols=["Id", "Stage"], index_col="Id")

    TcData2Result(tc_Stage=tc_Stage,
                  N_input=N_input, N_hidden=N_hidden, N_layers=N_layers, history_length=4,
                  memory_setting=memory_setting, train_setting=train_setting, epoch=1,
                  learn_rate=learn_rate, alpha=alpha, gamma=gamma,
                  train_flag=True, param_read=False, save_model_ori=True, save_model=True,
                  dataset_name=dataset_name, save_name=save_name)
    # 数据结果默认文件名格式 网络结构_网络层数_隐藏节点个数_数据集_网络学习率_alpha_gamma_epoch_reward-times
