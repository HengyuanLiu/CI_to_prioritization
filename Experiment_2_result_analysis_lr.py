import pandas as pd
import matplotlib.pyplot as plt
import os
import result_analysis
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
            lambda NAPFD_stage: result_analysis.tc_NAPFD(NAPFD_stage) if len(NAPFD_stage) >= self.mini_batch else -1
        ).reset_index()
        NAPFD_list.columns = ['Stage', 'NAPFD']
        x_y = NAPFD_list[NAPFD_list.NAPFD != -1]
        return x_y

    def NAPFD_sup(self):
        # 计算NAPFD曲线的上界
        NAPFD_list_max = self.data_grouped.apply(
            lambda NAPFD_stage: result_analysis.tc_NAPFD(NAPFD_stage, sort_by=['Verdict', 'Duration'],
                                                         ascending=[False, True]) if len(
                NAPFD_stage) >= self.mini_batch else -1
        ).reset_index()
        NAPFD_list_max.columns = ['Stage', 'NAPFD']
        x_y_max = NAPFD_list_max[NAPFD_list_max.NAPFD != -1]
        return x_y_max

    def NAPFD_inf(self):
        # 计算NAPFD曲线的下界
        NAPFD_list_min = self.data_grouped.apply(
            lambda NAPFD_stage: result_analysis.tc_NAPFD(NAPFD_stage, sort_by=['Verdict', 'Duration'],
                                                         ascending=[True, False]) if len(
                NAPFD_stage) >= self.mini_batch else -1
        ).reset_index()
        NAPFD_list_min.columns = ['Stage', 'NAPFD']
        x_y_min = NAPFD_list_min[NAPFD_list_min.NAPFD != -1]
        return x_y_min


load_path = r'./result_data/result_of_exp2/'
save_path_lr = r'./result_data/result_of_exp2/result_analysis_spec_lr'
save_path_alpha = r'./result_data/result_of_exp2/result_analysis_spec_alpha'
save_path_gamma = r'./result_data/result_of_exp2/result_analysis_spec_gamma'
if not os.path.exists(save_path_lr):
    os.makedirs(save_path_lr)

save_path = save_path_lr + '/'

alpha_list = [0.2, 0.5, 0.8]
gamma_list = [0.2, 0.5, 0.8]
learn_rate_list = [0.001, 0.01, 0.1]

# 将全部数据读取
data_dict = {}
for alpha in alpha_list:
    for gamma in gamma_list:
        for learn_rate in learn_rate_list:
            file_name = str(alpha) + '_' + str(gamma) + '_' + str(learn_rate) + '_exp2.csv'
            data_key = file_name[:-4]
            data_dict[data_key] = pd.read_csv(load_path + file_name, sep=';')

# 进行分析
for alpha in alpha_list:
    for gamma in gamma_list:
        # 数据进一步处理
        result_list = []  # [0.001, 0.01, 0.1]
        for learn_rate in learn_rate_list:
            data_key = str(alpha) + '_' + str(gamma) + '_' + str(learn_rate) + '_exp2'
            result_list.append(data_set(data_dict[data_key]))
        x_y_max = result_list[0].NAPFD_sup()
        x_y_min = result_list[0].NAPFD_inf()
        x_y_1000 = result_list[0].NAPFD_rank()
        x_y_100 = result_list[1].NAPFD_rank()
        x_y_10 = result_list[2].NAPFD_rank()

        index_list = (x_y_max.NAPFD > 1e-6) & (x_y_min.NAPFD < 1 - 1e-6)
        x_y_1000 = x_y_1000.loc[index_list, :]
        x_y_100 = x_y_100.loc[index_list, :]
        x_y_10 = x_y_10.loc[index_list, :]
        x_y_max = x_y_max.loc[index_list, :]
        x_y_min = x_y_min.loc[index_list, :]

        x = x_y_max.Stage.values
        x = x.reshape(-1, 1)

        y_1000 = x_y_1000.NAPFD.values
        y_1000 = y_1000.reshape(-1, 1)

        y_100 = x_y_100.NAPFD.values
        y_100 = y_100.reshape(-1, 1)

        y_10 = x_y_10.NAPFD.values
        y_10 = y_10.reshape(-1, 1)

        y_max = x_y_max.NAPFD.values
        y_max = y_max.reshape(-1, 1)

        y_min = x_y_min.NAPFD.values
        y_min = y_min.reshape(-1, 1)

        model_1000 = LinearRegression()
        model_100 = LinearRegression()
        model_10 = LinearRegression()
        model_max = LinearRegression()
        model_min = LinearRegression()

        model_1000.fit(x, y_1000)
        model_100.fit(x, y_100)
        model_10.fit(x, y_10)
        model_max.fit(x, y_max)
        model_min.fit(x, y_min)

        y_1000_pred = model_1000.predict(x)
        y_100_pred = model_100.predict(x)
        y_10_pred = model_10.predict(x)
        y_max_pred = model_max.predict(x)
        y_min_pred = model_min.predict(x)

        plt.ion()
        plt.plot(x, y_1000_pred, 'b--', label='lr='+str(0.001))
        plt.plot(x, y_100_pred, 'b-', label='lr='+str(0.01))
        plt.plot(x, y_10_pred, 'b-.', label='lr='+str(0.1))
        plt.plot(x, y_max_pred, 'r', label='NAPFD_max')
        plt.plot(x, y_min_pred, 'g', label='NAPFD_min')
        plt.legend()
        plt.xlabel('Stage')
        plt.ylabel('NAPFD')
        plt.title('alpha='+str(alpha)+' gamma='+str(gamma))
        save_name = str(alpha) + '_' + str(gamma) + '.jpg'
        plt.savefig(save_path+save_name)
        plt.pause(1)
        plt.ioff()
        plt.cla()
