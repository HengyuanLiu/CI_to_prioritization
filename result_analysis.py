import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import copy


class data_set:
    """排序结果数据集类"""

    def __init__(self, data, mini_batch=10):
        self.data = data
        self.data_grouped = data.groupby('Stage')
        self.mini_batch = mini_batch

    def NAPFD_rank(self):
        # 计算NAPFD的值
        NAPFD_list = self.data_grouped.apply(
            lambda NAPFD_stage: tc_NAPFD(NAPFD_stage) if len(NAPFD_stage) >= self.mini_batch else -1
        ).reset_index()
        NAPFD_list.columns = ['Stage', 'NAPFD']
        x_y = NAPFD_list[NAPFD_list.NAPFD != -1]
        return x_y

    def NAPFD_sup(self):
        # 计算NAPFD曲线的上界
        NAPFD_list_max = self.data_grouped.apply(
            lambda NAPFD_stage: tc_NAPFD(NAPFD_stage, sort_by=['Verdict', 'Duration'],
                                         ascending=[False, True]) if len(
                NAPFD_stage) >= self.mini_batch else -1
        ).reset_index()
        NAPFD_list_max.columns = ['Stage', 'NAPFD']
        x_y_max = NAPFD_list_max[NAPFD_list_max.NAPFD != -1]
        return x_y_max

    def NAPFD_inf(self):
        # 计算NAPFD曲线的下界
        NAPFD_list_min = self.data_grouped.apply(
            lambda NAPFD_stage: tc_NAPFD(NAPFD_stage, sort_by=['Verdict', 'Duration'],
                                         ascending=[True, False]) if len(
                NAPFD_stage) >= self.mini_batch else -1
        ).reset_index()
        NAPFD_list_min.columns = ['Stage', 'NAPFD']
        x_y_min = NAPFD_list_min[NAPFD_list_min.NAPFD != -1]
        return x_y_min


def data_std(y, y_max, y_min):
    y_std = copy.deepcopy(y)
    for i in range(len(y_std)):
        if y_max[i] - y_min[i] < 1e-6:
            if y_max[i] < 1e-6:
                y_std[i] = 1
            else:
                y_std[i] = y_max[i]
        else:
            y_std[i] = (y[i] - y_min[i]) / (y_max[i] - y_min[i])
    return y_std


def tc_NAPFD(sort_result, time_percentage=0.5, sort_by=['Rank'], ascending=[False]):
    """
    计算给定阶段的排序结果APFD
    输入数据至少应包含:[Verdict,Rank]
    """
    sort_result = sort_result.sort_values(by=sort_by, ascending=ascending)
    time_allow = time_percentage * sort_result.Duration.sum()
    allow_list = sort_result[sort_result.Duration.cumsum() <= time_allow]
    if sort_result.Verdict.sum() != 0:
        if allow_list.empty:
            NAPFD = 0
        else:
            NAPFD = allow_list.Verdict.cumsum().sum() / sort_result.Verdict.sum() / allow_list.shape[0] \
                    - allow_list.Verdict.sum() / sort_result.Verdict.sum() / allow_list.shape[0] / 2
            NAPFD = NAPFD.astype(dtype='float32')
    else:
        NAPFD = 1
    return NAPFD


def result_analysis(save_path=None, file_name=None, save_name='tc_result.csv', N_hidden=24, N_layers=3,
                    train_setting=None, memory_setting=None, dataset_name=None, show_flag=False, save_flag=True,
                    plot_NAPFD=True, plot_NAPFD_adj=True, mini_batch=10):
    if file_name is None:
        file_name = r"./result/result_data/result_of_" + dataset_name + '/' + save_name
    if dataset_name is None:
        dataset_name = 'iofrol'
    if memory_setting is None:
        memory_setting = 'current'
    if train_setting is None:
        train_setting = 'DNN'

    result = pd.read_csv(file_name, sep=';')
    result_groups = result.groupby("Stage")

    # 计算NAPFD的值
    NAPFD_list = result_groups.apply(
        lambda NAPFD_stage: tc_NAPFD(NAPFD_stage) if len(NAPFD_stage) >= mini_batch else -1
    ).reset_index()
    NAPFD_list.columns = ['Stage', 'NAPFD']
    x_y = NAPFD_list[NAPFD_list.NAPFD != -1]

    # 计算NAPFD曲线的上界
    NAPFD_list_max = result_groups.apply(
        lambda NAPFD_stage: tc_NAPFD(NAPFD_stage, sort_by=['Verdict', 'Duration'],
                                     ascending=[False, True]) if len(NAPFD_stage) >= mini_batch else -1
    ).reset_index()
    NAPFD_list_max.columns = ['Stage', 'NAPFD']
    x_y_max = NAPFD_list_max[NAPFD_list_max.NAPFD != -1]

    # 计算NAPFD曲线的下界
    NAPFD_list_min = result_groups.apply(
        lambda NAPFD_stage: tc_NAPFD(NAPFD_stage, sort_by=['Verdict', 'Duration'],
                                     ascending=[True, False]) if len(NAPFD_stage) >= mini_batch else -1
    ).reset_index()
    NAPFD_list_min.columns = ['Stage', 'NAPFD']
    x_y_min = NAPFD_list_min[NAPFD_list_min.NAPFD != -1]

    index_list = (x_y_max.NAPFD > 1e-6) & (x_y_min.NAPFD < 1 - 1e-6)
    x_y = x_y.loc[index_list, :]
    x_y_max = x_y_max.loc[index_list, :]
    x_y_min = x_y_min.loc[index_list, :]

    x = x_y.Stage.values
    x = x.reshape(-1, 1)

    y = x_y.NAPFD.values
    y = y.reshape(-1, 1)

    y_max = x_y_max.NAPFD.values
    y_max = y_max.reshape(-1, 1)

    y_min = x_y_min.NAPFD.values
    y_min = y_min.reshape(-1, 1)

    y_adj = data_std(y, y_max, y_min)

    model = LinearRegression()
    model_max = LinearRegression()
    model_min = LinearRegression()
    model_adj = LinearRegression()

    model.fit(x, y)
    model_max.fit(x, y_max)
    model_min.fit(x, y_min)
    model_adj.fit(x, y_adj)

    y_predict = model.predict(x)
    y_max_predict = model_max.predict(x)
    y_min_predict = model_min.predict(x)
    y_adj_predict = model_adj.predict(x)

    if plot_NAPFD:
        if show_flag:
            plt.ion()
        # plt.figure(1)
        plt.plot(x, y, 'b', label='NAPFD')
        plt.plot(x, y_max, 'r-.', label='NAPFD_max')
        plt.plot(x, y_min, 'g-.', label='NAPFD_min')

        plt.plot(x, y_predict, color='blue')
        plt.plot(x, y_max_predict, color='red')
        plt.plot(x, y_min_predict, color='green')
        plt.legend()
        plt.xlabel('Stage')
        plt.ylabel('NAPFD')
        jpg_name = save_name[0:-4] + '.jpg'
        if save_flag:
            if save_path is None:
                plt.savefig(r'./result/result_analysis/analysis_of_' + dataset_name + '/' + jpg_name)
            else:
                plt.savefig(save_path + '/' + jpg_name)
            plt.cla()
        if show_flag:
            plt.pause(0.1)
            plt.ioff()
            plt.cla()
            # plt.show()

    if plot_NAPFD_adj:
        if show_flag:
            plt.ion()
        # plt.figure(2)
        plt.plot(x, y_adj, label='NAPFD_adj')

        plt.plot(x, y_adj_predict)
        plt.legend()
        plt.xlabel('Stage')
        plt.ylabel('NAPFD')
        jpg_name_adj = save_name[0:-4] + '_adjustment.jpg'
        if save_flag:
            if save_path is None:
                plt.savefig(r'./result/result_analysis/analysis_of_' + dataset_name + '/' + jpg_name_adj)
            else:
                plt.savefig(save_path + '/' + jpg_name_adj)
            plt.cla()
        if show_flag:
            plt.pause(0.1)
            plt.ioff()
            plt.cla()
            # plt.show()


if __name__ == "__main__":
    dataset_name = "paintcontrol"
    # dataset_name = "iofrol"

    N_input = 6
    N_hidden = 24
    N_output = 1
    N_layers = 3
    history_length = 4
    train_setting = "DNN"
    memory_setting = "current"

    # save_name = train_setting + '_' + memory_setting + '_' + \
    #             str(N_layers) + '_' + str(N_hidden) + '_' + dataset_name + '_0.01_0.5_0.5.csv'
    data_save_path = './result_data/result_of_exp1'
    save_name = 'DQN_replay_paintcontrol_exp1.csv'
    result_analysis(file_name=data_save_path + '/' + save_name, save_name=save_name, N_hidden=N_hidden,
                    N_layers=N_layers,
                    train_setting=train_setting, memory_setting=memory_setting, dataset_name=dataset_name,
                    save_flag=True, show_flag=False, mini_batch=10)
