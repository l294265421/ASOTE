import numpy as np

from nlp_tasks.absa.utils import file_utils


def read_features(file_path):
    """

    Args:
        file_path: 文件路径，对应文件中每一行是一个样本，且由,分割，第二个元素是对应的特征
    Returns:
        list, 每个元素是对应样本的特征
    """
    lines = file_utils.read_all_lines(file_path)
    features = [line.split(',')[1] for line in lines]
    return features


def max_len(all_sample_features):
    """

    Args:
        all_sample_features: list of list of string
    Returns:
        所有子list的最大长度
    """
    result = 0
    for sample_features in all_sample_features:
        if len(sample_features) > result:
            result = len(sample_features)
    return result


def read_subject_of_sentiment_value(file_path):
    """

    Args:
        file_path: 文件路径，其中每行是一个样本，且第三个元素是样本的主题标签
    Returns:
        list, 每个元素是样本的主题标签
    """
    lines = file_utils.read_all_lines(file_path)
    features = [ line.split(',')[2] for line in lines]
    return features


def read_ids(file_path):
    """

    Args:
        file_path: 文件路径，每行一个样本，且第一个元素是样本的id
    Returns:
        list, 每个元素是样本的id
    """
    lines = file_utils.read_all_lines(file_path)
    ids = [line.split(',')[0] for line in lines]
    return ids


def read_subject_train_ids(file_path):
    """

    Args:
        file_path: 文件路径，每行一个样本，且第一个元素是样本的id
    Returns:
        list, 每个元素是样本的id
    """
    lines = file_utils.read_all_lines(file_path)
    ids = [line.split(',')[2] for line in lines]
    return ids


def read_field(file_path, field_index, separator=',', has_head=True):
    """

    Args:
        file_path: 文件路径，每行一个样本，且第一个元素是样本的id
    Returns:
        list, 每个元素是样本的id
    """
    lines = file_utils.read_all_lines(file_path)
    if has_head:
        lines = lines[1:]
    ids = [line.split(separator)[field_index] for line in lines]
    return ids


def read_labels(file_path):
    """

    Args:
        file_path: 文件路径，每行一个样本，且第一个元素是样本的label
    Returns:
        list, 每个元素是样本的label
    """
    lines = file_utils.read_all_lines(file_path)
    labels = [[int(label) for label in line.split(',')[0].split(' ')] for line in lines]
    return labels


def read_test_labels(file_path):
    """

    Args:
        file_path: 文件路径，每行一个样本
    """
    lines = file_utils.read_all_lines(file_path)
    labels = [[int(label) for label in line.split(',')[1:]] for line in lines[1:]]
    return labels


def repeat_element_in_list(list_of_element, list_of_len):
    """

    Args:
        list_of_element: 元素的list
        list_of_len: list_of_element中对应元素出现次数的list
    Returns:
        list，每一个元素是空格连接的list_of_len[i]个list_of_element[i]

    """
    assert len(list_of_element) == len(list_of_len)
    result = []
    for i in range(len(list_of_len)):
        repeated_element = []
        for j in range(list_of_len[i]):
            repeated_element.append(list_of_element[i])
        result.append(' '.join(repeated_element))
    return result


def read_feature_label(file_path):
    """

    Args:
         file_path:对应文件中每一行是一个样本，且由,分割，第一个元素是样本的label,
         第二个元素是对应的特征
    Returns:
        (特征list, label list)
    """
    X = read_features(file_path)
    y = read_labels(file_path)
    y = np.array(y)
    return X, y


def read_feature_id(file_path):
    """

    Args:
        file_path:文件路径，对应文件中每一行是一个样本，且由,分割，且第一个元素是样本的id,
        第二个元素是对应的特征,
    Returns:
        (特征list, id list)
    """
    X = read_features(file_path)
    id = read_ids(file_path)
    return X, id
