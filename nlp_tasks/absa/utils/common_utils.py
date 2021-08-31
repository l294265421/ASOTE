import numpy as np


def print_score(score: float, description: str=''):
    """

    :param score: precision、recall
    :param description
    """
    print('%s %.4f' % (description, score))


def print_mean_std(arr: list):
    #
    arr_mean = np.mean(arr)
    #
    arr_std = np.std(arr, ddof=1)
    print_score(arr_mean, '')
    print_score(arr_std, '')
