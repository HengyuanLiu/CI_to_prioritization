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

fig = plt.figure(1)
fig2=plt.figure(2)

std_list = []
res_aver_list = []

i=0
for train_setting in train_list:
    for memory_setting in memory_list:
        i+=1
        save_name = train_setting + '_' + memory_setting + '_' + dataset_name + '_exp1' + '.csv'

        data_path = data_save_path + '/' + save_name
        data = pd.read_csv(data_path, sep=';',index_col='Id')

        data_class = data_set(data)
        NAPFD = data_class.NAPFD_rank()
        model = LinearRegression()

        index_list = (data_class.NAPFD_sup().NAPFD > 1e-6) & (data_class.NAPFD_inf().NAPFD < 1 - 1e-6)
        NAPFD = NAPFD.loc[index_list, :]

        x = NAPFD.Stage.values
        x = x.reshape(-1, 1)

        y = NAPFD.NAPFD.values
        y = y.reshape(-1, 1)

        model.fit(x,y)

        y_pred = model.predict(x)

        std = NAPFD.NAPFD.std()
        res_std = (y-y_pred).std()
        std_aver = (y-y.mean()).mean()
        res_aver = (y - y_pred).mean()

        im = fig.add_subplot(4,1,i)
        im.plot(NAPFD.Stage,NAPFD.NAPFD-NAPFD.NAPFD.mean())
        im.set_xlabel('Stage')
        im.set_ylabel('NAPFD_std')
        im.set_title(train_setting+' '+memory_setting)

        im2 = fig2.add_subplot(4,1,i)
        im2.plot(x,y-y_pred)
        im2.set_xlabel('Stage')
        im2.set_ylabel('NAPFD_residual')
        im2.set_title(train_setting+' '+memory_setting)

        std_list.append([train_setting+'_'+memory_setting,y.mean(),std_aver, std])
        res_aver_list.append([train_setting+'_'+memory_setting, res_aver,res_std])
fig.savefig(result_analysis_path + '/' + 'exp1_std.jpg')
fig2.savefig(result_analysis_path + '/' + 'exp1_residual.jpg')

plt.cla()
std_list = pd.DataFrame(std_list)
res_aver_list = pd.DataFrame(res_aver_list)
std_list.to_excel(result_analysis_path + '/' + 'std_exp1.xlsx')
res_aver_list.to_excel(result_analysis_path + '/' + 'residual_exp1.xlsx')
