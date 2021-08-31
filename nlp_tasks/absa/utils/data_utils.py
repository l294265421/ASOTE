import numpy as np

from nlp_tasks.absa.utils import file_utils


def read_features(file_path):
    """

    Args:
        file_path: ，，,，
    Returns:
        list,
    """
    lines = file_utils.read_all_lines(file_path)
    features = [line.split(',')[1] for line in lines]
    return features


def max_len(all_sample_features):
    """

    Args:
        all_sample_features: list of list of string
    Returns:
        list
    """
    result = 0
    for sample_features in all_sample_features:
        if len(sample_features) > result:
            result = len(sample_features)
    return result


def read_subject_of_sentiment_value(file_path):
    """

    Args:
        file_path: ，，
    Returns:
        list,
    """
    lines = file_utils.read_all_lines(file_path)
    features = [ line.split(',')[2] for line in lines]
    return features


def read_ids(file_path):
    """

    Args:
        file_path: ，，id
    Returns:
        list, id
    """
    lines = file_utils.read_all_lines(file_path)
    ids = [line.split(',')[0] for line in lines]
    return ids


def read_subject_train_ids(file_path):
    """

    Args:
        file_path: ，，id
    Returns:
        list, id
    """
    lines = file_utils.read_all_lines(file_path)
    ids = [line.split(',')[2] for line in lines]
    return ids


def read_field(file_path, field_index, separator=',', has_head=True):
    """

    Args:
        file_path: ，，id
    Returns:
        list, id
    """
    lines = file_utils.read_all_lines(file_path)
    if has_head:
        lines = lines[1:]
    ids = [line.split(separator)[field_index] for line in lines]
    return ids


def read_labels(file_path):
    """

    Args:
        file_path: ，，label
    Returns:
        list, label
    """
    lines = file_utils.read_all_lines(file_path)
    labels = [[int(label) for label in line.split(',')[0].split(' ')] for line in lines]
    return labels


def read_test_labels(file_path):
    """

    Args:
        file_path: ，
    """
    lines = file_utils.read_all_lines(file_path)
    labels = [[int(label) for label in line.split(',')[1:]] for line in lines[1:]]
    return labels


def repeat_element_in_list(list_of_element, list_of_len):
    """

    Args:
        list_of_element: list
        list_of_len: list_of_elementlist
    Returns:
        list，list_of_len[i]list_of_element[i]

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
         file_path:，,，label,

    Returns:
        (list, label list)
    """
    X = read_features(file_path)
    y = read_labels(file_path)
    y = np.array(y)
    return X, y


def read_feature_id(file_path):
    """

    Args:
        file_path:，，,，id,
        ,
    Returns:
        (list, id list)
    """
    X = read_features(file_path)
    id = read_ids(file_path)
    return X, id
