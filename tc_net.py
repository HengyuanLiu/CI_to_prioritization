"""模块导入"""

import torch.nn as nn


"""自定义类"""


class Rank_net(nn.Module):
    """秩网络,给出测试用例在当前状态下的秩（即行动,对于测试用例位置的预测排序指标）"""

    def __init__(self, n_input=5, n_output=1, n_hidden=12):
        super(Rank_net, self).__init__()
        self.linear_start = nn.Linear(n_input, n_hidden)
        self.linear_end = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x1 = self.linear_start(x)
        y = self.linear_end(x1)
        return y


class Rank_net_active(nn.Module):
    """秩网络,给出测试用例在当前状态下的秩（即行动,对于测试用例位置的预测排序指标）"""

    def __init__(self, n_input=5, n_output=1, n_hidden=12):
        super(Rank_net_active, self).__init__()
        self.linear_start = nn.Linear(n_input, n_hidden)
        self.act1 = nn.ELU(alpha=0.25)
        self.linear_end = nn.Linear(n_hidden, n_output)
        self.actend = nn.ELU(alpha=0.25)

    def forward(self, x):
        x1 = self.linear_start(x)
        x1_a = self.act1(x1)
        y = self.linear_end(x1)
        y_a = self.actend(y)
        return y_a


class Rank_net_SixLayers(nn.Module):
    """秩网络,给出测试用例在当前状态下的秩（即行动,对于测试用例位置的预测排序指标）"""

    def __init__(self, n_input=5, n_output=1, n_hidden=12):
        super(Rank_net_SixLayers, self).__init__()
        self.linear_start = nn.Linear(n_input, n_hidden)
        self.linear_middle1 = nn.Linear(n_hidden, n_hidden)
        self.linear_middle2 = nn.Linear(n_hidden, n_hidden)
        self.linear_middle3 = nn.Linear(n_hidden, n_hidden)
        self.linear_end = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x1 = self.linear_start(x)
        x2 = self.linear_middle1(x1)
        x3 = self.linear_middle2(x2)
        x4 = self.linear_middle3(x3)
        y = self.linear_end(x4)
        return y


class Rank_net_NineLayers(nn.Module):
    """秩网络,给出测试用例在当前状态下的秩（即行动,对于测试用例位置的预测排序指标）"""

    def __init__(self, n_input=5, n_output=1, n_hidden=12):
        super(Rank_net_NineLayers, self).__init__()
        self.linear_start = nn.Linear(n_input, n_hidden)
        self.linear_middle1 = nn.Linear(n_hidden, n_hidden)
        self.linear_middle2 = nn.Linear(n_hidden, n_hidden)
        self.linear_middle3 = nn.Linear(n_hidden, n_hidden)
        self.linear_middle4 = nn.Linear(n_hidden, n_hidden)
        self.linear_middle5 = nn.Linear(n_hidden, n_hidden)
        self.linear_middle6 = nn.Linear(n_hidden, n_hidden)
        self.linear_end = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x1 = self.linear_start(x)
        x2 = self.linear_middle1(x1)
        x3 = self.linear_middle2(x2)
        x4 = self.linear_middle3(x3)
        x5 = self.linear_middle4(x4)
        x6 = self.linear_middle5(x5)
        x7 = self.linear_middle6(x6)
        y = self.linear_end(x7)
        return y


"""自定义函数"""


def initNetParams(net):
    """网络参数初始化"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, 0, 0.1)
            nn.init.constant_(m.bias, 0.1)
