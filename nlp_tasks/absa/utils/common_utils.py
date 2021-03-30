import numpy as np


def print_score(score: float, description: str=''):
    """

    :param score: precision、recall等评测分数
    :param description
    """
    print('%s %.4f' % (description, score))


def print_mean_std(arr: list):
    # 求均值
    arr_mean = np.mean(arr)
    # 求标准差
    arr_std = np.std(arr, ddof=1)
    print_score(arr_mean, '平均值')
    print_score(arr_std, '标准差')
