# -*- coding: utf-8 -*-


import sys
import json

from nlp_tasks.utils import file_utils

input_filepath = sys.argv[1]

accs = []
result_filepath_of_test = []
lines = file_utils.read_all_lines(input_filepath)
for line in lines:
    if 'atsa_train_templates.py-361' in line and 'data_type: test result' in line:
        start_index = line.index('{')
        performances_str = line[start_index:].replace('\'', '"')
        performances = json.loads(performances_str)
        accs.append(float('%.5f' % performances['sentiment_acc']))
        continue
    if 'result_of_predicting_tes' in line:
        start_index = line.index(':') + 1
        filepath = line[start_index:]
        result_filepath_of_test.append(filepath)
print('accs:')
print(','.join([str(e) for e in accs]))
print('filepaths: %s' % str(result_filepath_of_test))