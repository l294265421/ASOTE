import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from nlp_tasks.absa.conf import datasets, task_conf, thresholds
from nlp_tasks.absa.conf import model_output_type


def to_list(y):
    if isinstance(y, np.ndarray):
        y = y.tolist()
    return y


def to_1d_list(y):
    if isinstance(y, np.ndarray):
        y = y.tolist()
    result = []
    for one_sample in y:
        result += one_sample
    return result


def all_to_1d_list(y):
    if isinstance(y, np.ndarray):
        y = y.tolist()
    result = []
    for one_sample in y:
        result.append(to_1d_list(one_sample))
    return result


def all_to_list(y):
    if isinstance(y, np.ndarray):
        y = y.tolist()
    result = []
    for one_sample in y:
        result.append(to_list(one_sample))
    return result


def precision(y_true, y_score):
    t_p = tp(y_true, y_score)
    f_p = fp(y_true, y_score)
    print('tp+fp=%d' % (t_p + f_p))
    return t_p / (t_p + f_p + 0.000000001)


def recall(y_true, y_score):
    t_p = tp(y_true, y_score)
    f_n = fn(y_true, y_score)
    print('tp+fn=%d' % (t_p + f_n))
    result = t_p / (t_p + f_n + 0.000000001)
    return result


def f1(y_true, y_score):
    p = precision(y_true, y_score)
    r = recall(y_true, y_score)
    if p == 0 or r == 0:
        return 0.0
    else:
        return 2 * (p * r) / (p + r)


def tp(y_true, y_score):
    y_true = to_1d_list(y_true)
    y_score_int = to_1d_list(y_score)
    result = 0
    for i in range(len(y_true)):
        y_true_i = y_true[i]
        y_score_i = y_score_int[i]
        if y_true_i == y_score_i > 0:
            result += 1
    print('tp: %d' % result)
    return result


def fp(y_true, y_score):
    y_true = to_1d_list(y_true)
    y_score_int = to_1d_list(y_score)
    result = 0
    for i in range(len(y_true)):
        y_true_i = y_true[i]
        y_score_i = y_score_int[i]
        if 0 < y_score_i != y_true_i:
            result += 1
    print('fp: %d' % result)
    return result


def fn(y_true, y_score):
    y_true = to_1d_list(y_true)
    y_score_int = to_1d_list(y_score)
    result = 0
    for i in range(len(y_true)):
        y_true_i = y_true[i]
        y_score_i = y_score_int[i]
        if 0 < y_true_i != y_score_i:
            result += 1
    print('fn: %d' % result)
    return result


def judge_y_score(y_score, threshold):
    """

    :param y_score: list of list of float, 预测的概率
    :param threshold: list of float, y_score里的每个list中每个位置元素对应的概率
    :return: list of list of int, y_score中每个元素按照threshold转化为整数
    """

    result = []
    for i, one_predict in enumerate(y_score):
        # 应用于帮助预测概率里都没有超过阈值的情况
        max_prob = max(one_predict)
        predict_label = []
        for j, predict_probability in enumerate(one_predict):
            if predict_probability >= threshold[j] or predict_probability == max_prob:
                predict_label.append(1)
            else:
                predict_label.append(0)
        result.append(predict_label)
    return result


def to_multi_output_label(y):
    result = []
    for i in range(len(y[0])):
        one_label = y[:, i]
        result.append(np.array(one_label)[:, np.newaxis])
    return result


def to_normal_label(predict):
    predict_list = [to_list(e) for e in predict]
    result = []
    for i in range(len(predict_list[0])):
        one_sample = []
        for j in range(len(predict_list)):
            one_sample += predict_list[j][i]
        result.append(one_sample)
    return result


def vstack_of_ndarray_list(list_of_ndarray1, list_of_ndarray2):
    result = []
    for i in range(len(list_of_ndarray1)):
        result.append(np.vstack([list_of_ndarray1[i], list_of_ndarray2[i]]))
    return result


def to_normal_label_ndarray(predict):
    return np.array(to_normal_label(predict))


def one_hot_to_label(y):
    result = []
    for row in y:
        index = np.unravel_index(row.argmax(), row.shape)
        result.append(index[0])

    return np.array(result)


def end_to_end_lstm_to_normal_aspect_label(y_list: list) -> list:
    result = []
    for i in range(len(y_list[0])):
        result_element = []
        for j in range(len(y_list)):
            one_label = y_list[j][i]
            if one_label.index(max(one_label)) != task_conf.sentiment_class_num:
                result_element.append(1)
            else:
                result_element.append(0)
        result.append(result_element)
    return result


def evaluate_aspect_of_joint_model(y_pred, y_true, description, epoch):
    if nlp_tasks.conf.model_output_type == model_output_type.end_to_end_lstm:
        y_pred_list = all_to_list(y_pred)
        y_true_list = all_to_list(y_true)
        y_pred_aspect = end_to_end_lstm_to_normal_aspect_label(y_pred_list)
        y_true_aspect = end_to_end_lstm_to_normal_aspect_label(y_true_list)
    else:
        threshold = thresholds.topic_positive_threshold
        y_pred_aspect = judge_y_score(to_normal_label(y_pred[:task_conf.subject_class_num]),
                                      threshold)
        y_true_aspect = to_normal_label(y_true[:task_conf.subject_class_num])
    precision_tra = precision(y_true_aspect, y_pred_aspect)
    recall_tra = recall(y_true_aspect, y_pred_aspect)
    f1_tra = f1(y_true_aspect, y_pred_aspect)
    print("\n %s - epoch: %d - precision: %.4f - recall: %.4f - f1: %.4f \n"
          % (description, epoch + 1, precision_tra, recall_tra, f1_tra))
    return f1_tra


def ndarray_list_to_list_list(ndarray_list):
    return [array.tolist() for array in ndarray_list]


def find_label_by_max(list_of_probabilities):
    label = [e.index(max(e))for e in list_of_probabilities]
    return label


def convert_for_aspect_sentiment_f1(y_aspect, y_sentiment):
    result = []
    predict_aspect_count = 0
    y_aspect_normal = to_normal_label(y_aspect)
    for i, y_aspect_element in enumerate(y_aspect_normal):
        result_element = []
        max_prob = max(y_aspect_element)
        for j in range(len(y_aspect)):
            if y_aspect_element[j] >= thresholds.topic_positive_threshold[j] \
                    or (y_aspect_element[j] == max_prob != 0):
                predict_aspect_count += 1
                y_sentiment_ji = y_sentiment[j][i]
                max_index = y_sentiment_ji.index(max(y_sentiment_ji))
                result_element.append(max_index + 1)
            else:
                result_element.append(0)
        result.append(result_element)
    # print('predict_aspect_count: %d' % predict_aspect_count)
    return result


def end_to_end_lstm_to_to_normal_sentiment_aspect_label(y_list: list) -> list:
    result = []
    for i in range(len(y_list)):
        result_element = []
        for j in range(len(y_list[0])):
            one_label = y_list[i][j]
            if one_label.index(max(one_label)) != task_conf.sentiment_class_num:
                result_element.append([1])
            else:
                result_element.append([0])
        result.append(result_element)
    return result


def end_to_end_lstm_to_to_normal_sentiment_sentiment_label(y_list: list) -> list:
    result = []
    for i in range(len(y_list)):
        result_element = []
        for j in range(len(y_list[0])):
            one_label = y_list[i][j]
            result_element.append(one_label[:task_conf.sentiment_class_num])
        result.append(result_element)
    return result


def evaluate_sentiment_of_joint_model(y_pred, y_true, description, epoch):
    if nlp_tasks.conf.model_output_type == model_output_type.end_to_end_lstm:
        y_pred_list = all_to_list(y_pred)
        y_true_list = all_to_list(y_true)

        y_pred_aspect = end_to_end_lstm_to_to_normal_sentiment_aspect_label(y_pred_list)
        y_true_aspect = end_to_end_lstm_to_to_normal_sentiment_aspect_label(y_true_list)

        y_pred_sentiment = end_to_end_lstm_to_to_normal_sentiment_sentiment_label(y_pred_list)
        y_true_sentiment = end_to_end_lstm_to_to_normal_sentiment_sentiment_label(y_true_list)
    else:
        y_true_aspect = y_true[:task_conf.subject_class_num]
        y_true_aspect = ndarray_list_to_list_list(y_true_aspect)
        y_pred_aspect = y_pred[:task_conf.subject_class_num]
        y_pred_aspect = ndarray_list_to_list_list(y_pred_aspect)

        y_pred_sentiment = y_pred[task_conf.subject_class_num:]
        y_pred_sentiment = ndarray_list_to_list_list(y_pred_sentiment)
        y_true_sentiment = y_true[task_conf.subject_class_num:]
        y_true_sentiment = ndarray_list_to_list_list(y_true_sentiment)

    y_pred_sentiment_for_eval = convert_for_aspect_sentiment_f1(y_pred_aspect,
                                                                    y_pred_sentiment)
    y_true_sentiment_for_eval = convert_for_aspect_sentiment_f1(y_true_aspect, y_true_sentiment)
    precision_tra = precision(y_true_sentiment_for_eval, y_pred_sentiment_for_eval)
    recall_tra = recall(y_true_sentiment_for_eval, y_pred_sentiment_for_eval)
    f1_tra = f1(y_true_sentiment_for_eval, y_pred_sentiment_for_eval)

    y_pred_sentiment_for_eval = []
    y_true_sentiment_for_eval = []
    for i in range(len(y_true_aspect[0])):
        for j in range(len(y_true_aspect)):
            if y_true_aspect[j][i][0] == 1:
                y_pred_sentiment_for_eval.append(y_pred_sentiment[j][i])
                y_true_sentiment_for_eval.append(y_true_sentiment[j][i])
    acc = accuracy_score(find_label_by_max(y_true_sentiment_for_eval), find_label_by_max(y_pred_sentiment_for_eval))
    print("\n %s - epoch: %d - precision: %.4f - recall: %.4f - f1: %.4f - acc: %.4f \n"
          % (description, epoch + 1, precision_tra, recall_tra, f1_tra, acc))
    return acc, f1_tra
