# -*- coding: utf-8 -*-


import sys
import json
from collections import defaultdict

from nlp_tasks.utils import file_utils

input_filepath = sys.argv[1]


def print_precision_recall_f1(metrics: dict):
    """

    :param metrics:
    :return:
    """
    precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
    result = '\t'.join([','.join(precision), ','.join(recall), ','.join(f1)])
    print(result)


aste_metrics_all = defaultdict(list)
aspect_opinion_pair_metrics_all = defaultdict(list)
aspect_sentiment_pair_metrics_all = defaultdict(list)
atsa_metrics_all = []
towe_metrics_all = defaultdict(list)
result_filepaths_of_test = []
lines = file_utils.read_all_lines(input_filepath)
for line in lines:
    if 'sequence_labeling_train_templates.py-713' in line and 'data_type: test result' in line:
        start_index = line.index('{')
        performances_str = line[start_index:].replace('\'', '"')
        performances = json.loads(performances_str)

        aste_metrics = performances['aste_metrics']
        for key, value in aste_metrics.items():
            aste_metrics_all[key].append('%.5f' % value)

        aspect_opinion_pair_metrics = performances['aspect_term_opinion_term_metrics']
        for key, value in aspect_opinion_pair_metrics.items():
            aspect_opinion_pair_metrics_all[key].append('%.5f' % value)

        aspect_sentiment_pair_metrics = performances['aspect_term_sentiment_pair_metrics']
        for key, value in aspect_sentiment_pair_metrics.items():
            aspect_sentiment_pair_metrics_all[key].append('%.5f' % value)


        atsa_metrics_all.append('%.5f' % performances['sentiment_acc'])

        towe_metrics = {
            'precision': performances['precision'],
            'recall': performances['recall'],
            'f1': performances['f1']
        }
        for key, value in towe_metrics.items():
            towe_metrics_all[key].append('%.5f' % value)

        continue
    if 'result_of_predicting_tes' in line:
        start_index = line.index(':') + 1
        filepath = line[start_index:]
        result_filepaths_of_test.append(filepath)

print('aste_metrics_all:')
print_precision_recall_f1(aste_metrics_all)
print('aspect_opinion_pair_metrics:')
print_precision_recall_f1(aspect_opinion_pair_metrics_all)
print('aspect_sentiment_pair_metrics:')
print_precision_recall_f1(aspect_sentiment_pair_metrics_all)
print('sentiment:')
print(','.join(atsa_metrics_all))
print('towe_metrics:')
print_precision_recall_f1(towe_metrics_all)

# print('filepaths:')
# for filepath in result_filepaths_of_test:
#     print(filepath)