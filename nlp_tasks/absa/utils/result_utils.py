# -*- coding: utf-8 -*-
"""

Date:    2018/9/28 15:14
"""

import sys

import numpy as np

from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.conf import data_path, thresholds
from nlp_tasks.absa.utils import date_utils


def convert_sentiment_value_predict(predict_result):
    """

    Args:
        predict_result: list of list,每个元素都是概率值
    """
    if isinstance(predict_result, np.ndarray):
        predict_result = predict_result.tolist()
    result = []
    for one_predict_result in predict_result:
        predict_label = one_predict_result.index(max(one_predict_result))
        result.append(label_mapping.sentiment_value_mapping_reverse[str(predict_label)])
    return result


def convert_subject_predict(predict_result, threshold):
    """

    Args:
        predict_result: list of list,每个元素都是概率值
        threshold: float
    """
    if isinstance(predict_result, np.ndarray):
        predict_result = predict_result.tolist()
    result = []
    for one_predict_result in predict_result:
        subjects = []
        for i, value in enumerate(one_predict_result):
            if value > threshold[i]:
                subjects.append(label_mapping.subject_mapping_reverse[str(i)])

        if len(subjects) == 0:
            max_predict_index = one_predict_result.index(max(one_predict_result))
            subjects.append(label_mapping.subject_mapping_reverse[str(max_predict_index)])
        result.append('|'.join(subjects))
    return result


def convert_subject_sentiment_value_predict(predict_result, threshold=0.8):
    """

    Args:
        predict_result: list of list,每个元素都是概率值
        threshold: float
    """
    if isinstance(predict_result, np.ndarray):
        predict_result = predict_result.tolist()
    result = []
    for one_predict_result in predict_result:
        subjects = []
        for i, value in enumerate(one_predict_result):
            if value > threshold:
                subjects.append(label_mapping.subject_sentiment_value_mapping_reverse[str(i)])

        if len(subjects) == 0:
            max_predict_index = one_predict_result.index(max(one_predict_result))
            subjects.append(label_mapping.subject_sentiment_value_mapping_reverse[str(max_predict_index)])
        result.append('|'.join(subjects))
    return result


def merge(list1, list2):
    """merge"""
    result = [list1[i] + ',' + list2[i] for i in range(len(list1))]
    return result


def convert_predict_for_probability_output(y_pred):
    """convert_predict_for_probability_output"""
    y_pred_str = [['%.2f' % one_y_pred_element for one_y_pred_element in one_y_pred] for one_y_pred in y_pred]
    return [','.join(one_y_pred_str) for one_y_pred_str in y_pred_str]


def merge_subject_sentiment_value(subject_file_path, sentiment_file_path, result_file_path):
    """convert_subject_sentiment_value_predict_result"""
    subject_file_lines = file_utils.read_all_lines(subject_file_path)
    sentiment_file_lines = file_utils.read_all_lines(sentiment_file_path)
    result = ['content_id,subject,sentiment_value,sentiment_word']
    for i, subject_line in enumerate(subject_file_lines):
        subject_line_parts = subject_line.split(',')

        sentiment_value = sentiment_file_lines[i].split(',')[1]

        result.append(subject_line_parts[0] + ',' + subject_line_parts[2] + ',' + sentiment_value + ',')

    file_utils.write_lines(result, result_file_path)


def convert_subject_sentiment_value_predict_result(subject_subject_sentiment_value_file_path, result_file_path):
    """convert_subject_sentiment_value_predict_result"""
    subject_sentiment_value_file_lines = file_utils.read_all_lines(subject_subject_sentiment_value_file_path)
    result = ['content_id,subject,sentiment_value,sentiment_word']
    for i, subject_line in enumerate(subject_sentiment_value_file_lines):
        id_subjects = subject_line.split(',')
        subjects = id_subjects[1].split('|')
        for subject in subjects:
            result.append(id_subjects[0] + ',' + subject.replace('_', ',') + ',')
    file_utils.write_lines(result, result_file_path)


def save_sentiment_value_result(y_pred, id_test, model_name, is_val=False):
    """

    Args:
        y_pred: 预测结果
        id_test: 测试集中所有id,保持顺序
    """
    if y_pred is None:
        return
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    y_pred_probability = convert_predict_for_probability_output(y_pred)
    id_probabilities = merge(id_test, y_pred_probability)
    head = 'id,' + ','.join(label_mapping.sentiment_value_mapping_list)
    if is_val:
        file_utils.write_lines([head] + id_probabilities,
                               data_path.val_sentiment_value_probability_result_file_path + '.'
                               + model_name)
    else:
        file_utils.write_lines([head] + id_probabilities,
                               data_path.test_public_sentiment_value_probability_result_file_path + '.'
                               + model_name)

    predict_labels = convert_sentiment_value_predict(y_pred)

    id_labels = merge(id_test, predict_labels)

    if is_val:
        file_utils.write_lines(id_labels, data_path.val_sentiment_value_result_file_path)
    else:
        file_utils.write_lines(id_labels, data_path.test_public_sentiment_value_result_file_path)


def save_sentiment_value_result_for_topic(y_pred, id_test, model_name, topic):
    """

    Args:
        y_pred: 预测结果
        id_test: 测试集中所有id,保持顺序
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    y_pred_probability = convert_predict_for_probability_output(y_pred)
    id_probabilities = merge(id_test, y_pred_probability)
    head = 'id,' + ','.join(label_mapping.sentiment_value_mapping_list)
    file_utils.write_lines([head] + id_probabilities,
                           data_path.test_public_sentiment_value_probability_result_file_path + '.'
                           + model_name)

    predict_labels = convert_sentiment_value_predict(y_pred)

    id_labels = merge(id_test, predict_labels)

    file_utils.write_lines(id_labels,
                           data_path.test_public_sentiment_value_result_file_path + '.' + topic)


sentiment_onehot_label = {
    '1 0 0': '-1',
    '0 1 0': '0',
    '0 0 1': '1'
}


def save_train_sentiment_value_result(y_pred, y_data, out_file_path):
    """

    Args:
        y_pred: 预测结果
        id_test: 测试集中所有id,保持顺序
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()

    predict_labels = convert_sentiment_value_predict(y_pred)

    id_labels = merge(y_data, predict_labels)

    result = []
    for i, id_label in enumerate(id_labels):
        parts = id_label.split(',')
        p = int(parts[-1]) + 1
        a = [e for e in parts[0].split()]
        pred_p = [('%.2f' % num) for num in y_pred[i]]
        parts.append(' '.join(pred_p))
        temp = [sentiment_onehot_label[parts[0]], parts[4], parts[5], parts[2], parts[1], parts[3]]
        parts = temp
        if a[p] != '1':
            result.append(','.join(parts))
        else:
            result.insert(0, ','.join(parts))
    result = ['情感,预测情感,预测概率,主题,内容,id'] + result
    file_utils.write_lines(result, out_file_path)


def save_subject_result(y_pred, id_test, model_name, is_val=False):
    """save_subject_result"""
    if y_pred is None:
        return
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    y_pred_probability = convert_predict_for_probability_output(y_pred)
    id_probabilities = merge(id_test, y_pred_probability)
    head = 'id,' + ','.join(label_mapping.subject_mapping_list)
    if is_val:
        file_utils.write_lines([head] + id_probabilities,
                               data_path.val_subject_probability_result_file_path + '.' +
                               model_name)
    else:
        file_utils.write_lines([head] + id_probabilities,
                               data_path.test_subject_probability_result_file_path + '.' +
                               model_name)

    predict_labels = convert_subject_predict(y_pred, threshold=thresholds.topic_positive_threshold)

    id_labels = merge(id_test, predict_labels)

    if is_val:
        file_utils.write_lines(id_labels, data_path.val_subject_result_file_path)
    else:
        file_utils.write_lines(id_labels, data_path.test_subject_result_file_path)


def save_train_subject_result(y_pred, y_data, model_name):
    """save_subject_result"""
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()

    predict_labels = convert_subject_predict(y_pred, threshold=thresholds.topic_positive_threshold)

    id_labels = merge(y_data, predict_labels)

    y_true = [[float(p) for p in data.split(',')[0].split()] for data in y_data]
    true_labels = convert_subject_predict(y_true, threshold=thresholds.topic_positive_threshold)

    y_pred_probability = convert_predict_for_probability_output(y_pred)
    result = []
    result.append('主题,预测主题,预测概率,内容,id')
    for i, id_label in enumerate(id_labels):
        if true_labels[i] == predict_labels[i]:
            continue
        parts = id_label.split(',')
        parts[0] = true_labels[i]

        pred_p = y_pred_probability[i]
        pred_p_elements = pred_p.split(',')
        pred_p_str_list = []
        for j in range(len(pred_p_elements)):
            label = label_mapping.subject_mapping_reverse[str(j)]
            pred_p_str_list.append(label + ':' + pred_p_elements[j])
        parts.insert(1, ' '.join(pred_p_str_list))
        parts.insert(1, parts[-1])
        del parts[-1]
        result.append(','.join(parts))

    result.sort()
    file_utils.write_lines(result, data_path.train_subject_result_file_path)


if __name__ == '__main__':
    merge_subject_sentiment_value(data_path.test_public_for_sentiment_value_exact_word_file_path,
                                  data_path.test_public_sentiment_value_result_file_path,
                                  data_path.test_public_result_file_path + '_' + sys.argv[1] + '.csv')
    # convert_subject_sentiment_value_predict_result(data_path.test_public_subject_sentiment_value_result_file_path,
    #                                                data_path.test_public_result_file_path + '_' + date_utils.now() + '.csv')
