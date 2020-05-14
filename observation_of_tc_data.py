import random
import pandas as pd
import matplotlib.pyplot as plt
import copy
import tc_data_reconstitution as tcDataRecon


def tc_data_cycle_read(tc_data_input, file_name, cycle):
    """
    根据给定的集成轮次指标读取数据
    输入数据至少包含[Cycle]数据
    """
    index_list = list(tc_data_input[tc_data_input.Cycle != cycle].index)
    tc_data = pd.read_csv(file_name,
                          sep=';', index_col="Id",
                          skiprows=index_list)
    return tc_data


def tc_data_stage_newproportion(tc_data_input, DataSetName, plot_show=False):
    """计算数据集中各轮次新增测试用例个数"""
    x = list(range(tc_data_input.Stage.min(), tc_data_input.Stage.max() + 1))
    tc_data = copy.deepcopy(tc_data_input)
    tc_data.reindex(columns=list(tc_data.columns) + ['times'], fill_value=1)
    for name in list(set(tc_data.Name)):
        tc_data.loc[tc_data.Name == name, 'times'] = tc_data.loc[tc_data.Name == name, 'times'].cumsum()
    y = []
    for stage in x:
        NewProportion = tc_data[(tc_data.Stage == stage) & (tc_data.times == 1)].shape[0] / \
                        tc_data[(tc_data.Stage == stage)].shape[0]
        y.append(NewProportion)
    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel('stage')
    plt.ylabel('proportion')
    save_path = './result/observation_of_tc_data/observation_of_' + DataSetName
    save_name = 'new test cases proportion each stage of ' + DataSetName
    plt.title(save_name)
    if plot_show:
        plt.show()
    plt.savefig(save_path + '/' + save_name)


def tc_data_cycle_sca(dataset_name, tc_data_input, cycle=1, color_name=None):
    """用于测试用例的观察函数"""
    # 数据分别是该测试用例在本轮中该测试用例第几次运行，一共运行几次，以及判定的颜色
    if color_name is None:
        color_name = ['Color_time', 'Color_times', 'Color_verdict']
    tc_cycle = tc_data_input[tc_data_input.Cycle == cycle]
    tc_cycle = tc_cycle.reindex(columns=list(tc_cycle.columns) + color_name, fill_value='')
    # 生成测试次数
    for tc_name in list(set(tc_cycle.Name)):
        times = len(list(tc_cycle.loc[tc_cycle.Name == tc_name, 'Name']))
        tc_cycle.loc[tc_cycle.Name == tc_name, 'Color_times'] = times
        tc_cycle.loc[tc_cycle.Name == tc_name, 'Color_time'] = list(range(1, times + 1))

    # 生成判定颜色
    tc_cycle.loc[tc_cycle.Verdict == 0, 'Color_verdict'] = 'k'
    tc_cycle.loc[tc_cycle.Verdict == 1, 'Color_verdict'] = 'r'

    # 绘图部分
    fig = plt.figure(1)
    sca1 = fig.add_subplot(3, 1, 1)
    sca2 = fig.add_subplot(3, 1, 2)
    sca3 = fig.add_subplot(3, 1, 3)

    for color in list(set(list(tc_cycle.Color_times))):
        sca1.scatter(list(tc_cycle[tc_cycle.Color_times == color].index),
                     list(tc_cycle[tc_cycle.Color_times == color].Name),
                     label=color)
    for color in list(set(list(tc_cycle.Color_time))):
        sca2.scatter(list(tc_cycle[tc_cycle.Color_time == color].index),
                     list(tc_cycle[tc_cycle.Color_time == color].Name),
                     label=color)
    for color in list(set(list(tc_cycle.Color_verdict))):
        if color == 'r':
            flag = 'failed'
        elif color == 'k':
            flag = 'pass'
        else:
            flag = 'error'
        sca3.scatter(list(tc_cycle[tc_cycle.Color_verdict == color].index),
                     list(tc_cycle[tc_cycle.Color_verdict == color].Name),
                     label=flag)

    sca1.set_xlabel('tc_index')
    sca2.set_xlabel('tc_index')
    sca3.set_xlabel('tc_index')

    sca1.set_ylabel('tc_name')
    sca2.set_ylabel('tc_name')
    sca3.set_ylabel('tc_name')

    sca1.legend(loc='upper left')
    sca2.legend(loc='upper left')
    sca3.legend(loc='upper left')
    title = './result/observation_of_tc_data/observation_of_' + dataset_name + \
            '/observation of ' + dataset_name + ' scatter_cycle ' + str(cycle)
    # plt.title(title)
    plt.savefig(title + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def tc_Stage_number_view(tc_data_input, dataset_name):
    """
    观察在阶段重构后各集成阶段的测试用例个数阶段
    输入数据仅需包含[LastRun]
    """
    tc_grouped = tc_data.groupby('Stage').count().iloc[:, 0]
    tc_grouped = tc_grouped.reset_index().drop(columns=['Stage'])
    tc_grouped.rename(columns={'Name': 'Stage'}, inplace=True)
    plt.figure(2)
    plt.bar(tc_grouped.index, tc_grouped.Stage)
    plt.xlim([tc_grouped.index.min() - 10, tc_grouped.index.max() + 10])
    plt.ylim([tc_grouped.Stage.min(), tc_grouped.Stage.max() + 10])
    title = './result/observation_of_tc_data/observation_of_' + dataset_name + \
            '/observation of ' + dataset_name + ' bar'
    plt.xlabel('Stage')
    plt.ylabel('Number of Test Cases')
    plt.savefig(title + '.png')
    # plt.show()
    plt.clf()


def tc_observation(dataset_name, file_name):
    # 读取数据
    tc_data = pd.read_csv(file_name, sep=';', index_col="Id")
    # 数据的描述性分析

    # print(tc_data.describe())
    # print(tc_data.notnull().any())  # 判断是否有空值
    # print(tc_data.columns)  # 查看各行名称

    if dataset_name == 'paintcontrol':
        tc_Stage_number_view(tc_data, stage_flag='Cycle')  # paintcontrol适合用'Cycle'作为分阶段指标
    elif dataset_name == 'iofrol':
        tc_Stage_number_view(tc_data, stage_flag='LastRun')  # iofrol适合用'LastRun'作为分阶段指标

    tc_data_cycle = pd.read_csv(file_name, sep=';', index_col="Id", usecols=["Id", "Cycle"])

    cycle_sample = []
    while len(cycle_sample) < 10:
        cycle = random.randint(tc_data_cycle.min(), tc_data_cycle.max())
        if tc_data_cycle.loc[tc_data_cycle.Cylce == cycle, :].shape[0] >= 10:
            cycle_sample.append(cycle)
    for cycle in cycle_sample:
        tc_data_cycle_input = tc_data_cycle_read(tc_data_cycle, file_name, cycle)
        tc_data_cycle_sca(dataset_name, tc_data_cycle_input, cycle)


if __name__ == '__main__':
    dataset_name = "paintcontrol"
    # dataset_name = "iofrol"
    file_name = r"./data/" + dataset_name + ".csv"
    # 读取数据
    tc_data = pd.read_csv(file_name, sep=';', index_col="Id")
    # 数据的描述性分析

    # print(tc_data.describe())
    # print(tc_data.notnull().any())  # 判断是否有空值
    # print(tc_data.columns)  # 查看各行名称

    tc_Stage_number_view(tc_data, dataset_name)

    tc_data_cycle = pd.read_csv(file_name, sep=';', index_col="Id", usecols=["Id", "Cycle"])

    cycle_sample = []
    while len(cycle_sample) < 10:
        cycle = random.randint(tc_data_cycle.Cycle.min(), tc_data_cycle.Cycle.max())
        if tc_data_cycle.loc[tc_data_cycle.Cycle == cycle, :].shape[0] >= 10:
            cycle_sample.append(cycle)
    for cycle in cycle_sample:
        tc_data_cycle_input = tc_data_cycle_read(tc_data_cycle, file_name, cycle)
        tc_data_cycle_sca(dataset_name, tc_data_cycle_input, cycle)
