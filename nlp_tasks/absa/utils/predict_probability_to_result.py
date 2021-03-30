# -*- coding: utf-8 -*-


from nlp_tasks.absa.conf import task_conf
from nlp_tasks.absa.utils import evaluate_utils
from nlp_tasks.absa.preprocess import label_mapping


def acaoa_predict_probability_to_result(predict_probability):
    """

    :return:
    """
    y_aspect = predict_probability[:task_conf.subject_class_num]
    y_aspect = evaluate_utils.ndarray_list_to_list_list(y_aspect)

    y_sentiment = predict_probability[task_conf.subject_class_num:]
    y_sentiment = evaluate_utils.ndarray_list_to_list_list(y_sentiment)

    result = predict_probability_to_result(y_aspect,  y_sentiment)
    return result


def acaoa_predict_probability_to_result_all_sentiment(predict_probability, true_probability):
    """

    :return:
    """
    y_aspect = true_probability[:task_conf.subject_class_num]
    y_aspect = evaluate_utils.ndarray_list_to_list_list(y_aspect)

    y_sentiment = predict_probability[task_conf.subject_class_num:]
    y_sentiment = evaluate_utils.ndarray_list_to_list_list(y_sentiment)

    result = predict_probability_to_result(y_aspect, y_sentiment)
    return result


def predict_probability_to_result(y_aspect, y_sentiment):
    """

    :return:
    """
    y = evaluate_utils.convert_for_aspect_sentiment_f1(y_aspect, y_sentiment)
    result = []
    for example_predict in y:
        example_predict_label = []
        for i, aspect_predict in enumerate(example_predict):
            if aspect_predict == 0:
                continue
            aspect = label_mapping.subject_mapping_reverse[str(i)]
            sentiment = label_mapping.sentiment_value_mapping_reverse[str(aspect_predict - 1)]
            example_predict_label.append([aspect, sentiment])
        result.append(example_predict_label)
    return result


def end_to_end_lstm_probability_to_result():
    """

    :return:
    """
    pass
