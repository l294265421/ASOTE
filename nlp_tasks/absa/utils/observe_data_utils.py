import json

from nlp_tasks.absa.utils import predict_probability_to_result
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.conf import task_conf, data_path


def observe_acaoa_aspect_sentiment_pair(test_text, y_test, y_pred, model_name, all_sentiment=False):
    if all_sentiment:
        predict_label = predict_probability_to_result.acaoa_predict_probability_to_result_all_sentiment(y_pred, y_test)
    else:
        predict_label = predict_probability_to_result.acaoa_predict_probability_to_result(y_pred)
    true_label = predict_probability_to_result.acaoa_predict_probability_to_result(y_test)
    observe_data = []
    for i in range(len(predict_label)):
        observe_data.append([test_text[i], true_label[i], predict_label[i]])
    json.dump(observe_data,
              open(data_path.data_base_dir + ('observe_data_%s.json' % model_name),
                   encoding='utf-8',
                   mode='w'),
              ensure_ascii=False)