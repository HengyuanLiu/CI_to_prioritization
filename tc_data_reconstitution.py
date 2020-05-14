import pandas as pd
import datetime
import random


def tc_data_generalize_Stage(tc_data_input):
    """
    生成阶段标签
    输入DataFrame仅需包含['Name','Cycle']
    """
    tc_data = tc_data_input.reindex(columns=list(tc_data_input.columns) + ['Stage'], fill_value=1)
    stage = 1
    for cycle in range(tc_data.index.min(), tc_data.index.max()):
        tc_list = []
        tc_data_cycle = tc_data[tc_data.Cycle == cycle]
        for index in list(tc_data_cycle.index):
            tc_name = tc_data_cycle.loc[index, 'Name']
            if tc_name in tc_list:
                tc_list = []
                stage += 1
            else:
                tc_list.append(tc_name)
            tc_data.loc[index, 'Stage'] = stage
        stage += 1
    tc_data.Stage = tc_data.Stage.astype(dtype=int)
    return tc_data


def tc_data_generalize_FreeTime(tc_data_input, tc_Name, filename):
    """
    计算该测试用例的空闲时间
    输入数据至少包含[Name,LastRun]
    """
    tc_data_input = tc_data_input.reindex(columns=list(tc_data_input.columns)+['FreeTime'], fill_value=0)
    for index in list(tc_data_input.index):
        # 读取已经当前测试用例的上次执行时间
        tc_name = tc_data_input.loc[index, 'Name']
        index_last = list(tc_Name[(tc_Name.Name == tc_name) & (tc_Name.index < index)].index)
        if index_last:
            index_last = index_last[-1]
            index_list = list(tc_Name[tc_Name.index != index_last].index)
            tc_last = pd.read_csv(filename, sep=";", index_col="Id", skiprows=index_list)

            d1 = datetime.datetime.strptime(tc_last.loc[index_last, 'LastRun'], '%Y-%m-%d %H:%M:%S')
            d2 = datetime.datetime.strptime(tc_data_input.loc[index, 'LastRun'], '%Y-%m-%d %H:%M:%S')
            delta = d2 - d1

            tc_data_input.loc[index, 'FreeTime'] = delta.days + delta.seconds / 86400
            # 将位置信息数据转为float类型
            tc_data_input.FreeTime = tc_data_input.FreeTime.astype(dtype='float')
    return tc_data_input


def tc_data_generalize_RunTime(tc_data_input, tc_Name, filename, run_time_average=0, run_time_std=0):
    """
    计算该测试用例的空闲时间
    输入数据至少包含[Name,LastRun]
    """
    tc_data_input = tc_data_input.reindex(columns=list(tc_data_input.columns)+['RunTime'], fill_value=0)
    for index in list(tc_data_input.index):
        # 读取已经当前测试用例的上次执行时间
        tc_name = tc_data_input.loc[index, 'Name']
        index_last = list(tc_Name[(tc_Name.Name == tc_name)&(tc_Name.index < index)].index)
        if index_last:
            index_last = index_last[-1]
            index_list = list(tc_Name[tc_Name.index != index_last].index)
            tc_last = pd.read_csv(filename, sep=";", index_col="Id", skiprows=index_list)

            tc_data_input.loc[index, 'RunTime'] = tc_last.loc[index_last, 'Duration']
            # 将位置信息数据转为float类型
            tc_data_input.RunTime = tc_data_input.RunTime.astype(dtype='float')
        else:
            if run_time_average > 1e-6:
                tc_data_input.loc[index, 'RunTime'] = random.normalvariate(run_time_average, run_time_std)
            else:
                pass
    return tc_data_input


def tc_LastResults_split(tc_data_input, history_length=4):
    """
    将历史日志的数据且分为易读数据
    输入数据至少含有['LastResults']
    """
    # print('-' * 50)
    # print("开始且分为易输入数据~")
    tc_last_in = list(map(eval, tc_data_input.LastResults))
    for index_tc in range(len(tc_last_in)):
        while len(tc_last_in[index_tc]) != history_length:
            if len(tc_last_in[index_tc]) < history_length:
                tc_last_in[index_tc].insert(0, 1)
            elif len(tc_last_in[index_tc]) > history_length:
                tc_last_in[index_tc] = tc_last_in[index_tc][0:history_length]
        #     print(tc_last_in[index_tc])
        # print('_' * 50)

    tc_data_input = tc_data_input.reindex(
        columns=list(tc_data_input.columns) +
                [('LastResult' + str(R_name)) for R_name in range(1, history_length + 1)],
        fill_value=1)
    tc_data_input.iloc[:, -history_length:] = tc_last_in
    # print("数据切分完毕,哔---")
    return tc_data_input


def tc_data_stage_read(tc_data_stage, stage, filename):
    """
    根据给定的集成阶段指标读取数据
    输入DataFrame至少包括['Stage']
    """
    index_list = list(tc_data_stage[tc_data_stage.Stage != stage].index)
    tc_data = pd.read_csv(filename,
                          sep=';', index_col="Id",
                          skiprows=index_list)
    return tc_data
