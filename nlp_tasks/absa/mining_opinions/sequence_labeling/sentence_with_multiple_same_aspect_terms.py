# -*- coding: utf-8 -*-


import argparse
import sys
import random
import copy
from typing import List
import json
from collections import defaultdict
import os
import traceback

import torch
import numpy

from nlp_tasks.absa.utils import argument_utils
from nlp_tasks.absa.mining_opinions.data_adapter import data_object
from nlp_tasks.utils import file_utils
from nlp_tasks.common import common_path

parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='ASOTEDataLapt14', type=str)
parser.add_argument('--version', help='data version', default='v2', type=str)
args = parser.parse_args()

configuration = args.__dict__

dataset_name = configuration['current_dataset']
dataset = data_object.get_dataset_class_by_name(dataset_name)(configuration)
train_dev_test_data = dataset.get_data_type_and_data_dict()
test_data = train_dev_test_data['test']
sentence_and_aspect_terms = defaultdict(list)
for sample in test_data:
    sentence = sample.text
    try:
        aspect = sample.metadata['original_line_data']['aspect_term']['term']
        sentence_and_aspect_terms[sentence].append(aspect)
    except:
        print(traceback.format_exc())
for sentence, aspect_terms in sentence_and_aspect_terms.items():
    aspect_counter = defaultdict(int)
    for aspect in aspect_terms:
        aspect_counter[aspect] += 1
    for aspect, counter in aspect_counter.items():
        if counter > 1:
            print('%s %s' % (sentence, aspect))


