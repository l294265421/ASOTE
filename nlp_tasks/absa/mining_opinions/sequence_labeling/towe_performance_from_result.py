# -*- coding: utf-8 -*-


import sys
import json

from nlp_tasks.utils import file_utils

input_filepath = sys.argv[1]

precisions = []
recalls = []
f1s = []
result_filepath_of_test = []
lines = file_utils.read_all_lines(input_filepath)
for line in lines:
    if 'sequence_labeling_train_templates.py-456' in line and 'data_type: test result' in line:
        start_index = line.index('{')
        performances_str = line[start_index:].replace('\'', '"')
        performances = json.loads(performances_str)
        precisions.append(str('%.3f' % (performances['precision'] * 100)))
        recalls.append(str('%.3f' % (performances['recall'] * 100)))
        f1s.append(str('%.3f' % (performances['f1'] * 100)))
        continue
    if 'result_of_predicting_tes' in line:
        start_index = line.index(':') + 1
        filepath = line[start_index:]
        result_filepath_of_test.append(filepath)
print('precisions:')
print(','.join(precisions))
print('recalls:')
print(','.join(recalls))
print('f1s:')
print(','.join(f1s))
print('p r f')
print('\t'.join([','.join(precisions), ','.join(recalls), ','.join(f1s)]))
print('filepaths: %s' % str(result_filepath_of_test))