"""奖励函数库"""


def tc_reward(verdict):
    """单个测试用例的奖励函数"""
    return verdict


def tc_reward_balance(verdict):
    """单个测试用例的奖励函数"""
    return verdict - 0.5


def reward_function(verdict, function_name='tc_reward'):
    """奖励函数"""
    if function_name == 'tc_reward':
        return tc_reward(verdict)
    elif function_name == 'tc_reward_balance':
        return tc_reward_balance(verdict)
