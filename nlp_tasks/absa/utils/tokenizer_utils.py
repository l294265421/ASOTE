# -*- coding: utf-8 -*-
"""

Date:    2018/9/28 15:14
"""

import os
import warnings
import json

import numpy as np
from keras.preprocessing import text

from nlp_tasks.absa.conf import data_path

np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '8'


def get_tokenizer(topic_or_sentiment='topic', word_or_char='word'):
    """返回固定的tokenizer"""
    tokenizer = text.Tokenizer(lower=False, filters='')
    if topic_or_sentiment == 'sentiment':
        if word_or_char == 'char':
            word_index = json.load(open(data_path.data_base_dir + data_path.
                                        char_index_sentiment_file_path,
                                        encoding='utf-8'))
        else:
            word_index = json.load(open(data_path.data_base_dir + data_path.
                                        word_index_sentiment_file_path,
                                        encoding='utf-8'))
    else:
        word_index = json.load(open(data_path.word_index_subject_file_path, encoding='utf-8'))

    tokenizer.word_index = word_index
    return tokenizer
