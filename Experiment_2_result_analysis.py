import os
import pandas as pd
import priorization
import result_analysis
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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


train_setting = 'DQN'
memory_setting = 'current'

alpha_list = [0.2, 0.5, 0.8]
gamma_list = [0.2, 0.5, 0.8]
learn_rate_list = [0.001, 0.01, 0.1]
data_save_path = './result_data/result_of_exp2'
params_path = data_save_path + '/params'
result_analysis_path = data_save_path + '/result_analysis'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

fig = plt.figure(1)
fig2 = plt.figure(2)

std_list = []

i = 0
for alpha in alpha_list:
    for gamma in gamma_list:
        for learn_rate in learn_rate_list:
            i += 1
            save_name = str(alpha) + '_' + str(gamma) + '_' + str(learn_rate) + '_exp2' + '.csv'

            data_path = data_save_path + '/' + save_name
            data = pd.read_csv(data_path, sep=';', index_col='Id')

            data_class = data_set(data)
            NAPFD = data_class.NAPFD_rank()
            model = LinearRegression()

            index_list = (data_class.NAPFD_sup().NAPFD > 1e-6) & (data_class.NAPFD_inf().NAPFD < 1 - 1e-6)
            NAPFD = NAPFD.loc[index_list, :]

            x = NAPFD.Stage.values
            x = x.reshape(-1, 1)

            y = NAPFD.NAPFD.values
            y = y.reshape(-1, 1)

            model.fit(x, y)

            y_pred = model.predict(x)

            std = NAPFD.NAPFD.std()
            res_std = (y - y_pred).std()
            std_aver = (y - y.mean()).mean()
            res_aver = (y - y_pred).mean()

            std_list.append([str(alpha) + '_' + str(gamma) + '_' + str(learn_rate), y.mean(), std_aver, std, res_std])

std_list = pd.DataFrame(std_list)
std_list.to_excel(result_analysis_path + '/' + 'std_exp2.xlsx')
