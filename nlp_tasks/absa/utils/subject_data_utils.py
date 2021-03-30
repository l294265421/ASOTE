import re

import numpy as np
from keras.preprocessing import sequence

from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.conf import datasets


def generate_x(file_path, tokenizer, maxlen):
    x = data_utils.read_field(file_path, 1, separator=datasets.delimeter)
    x = tokenizer.texts_to_sequences(x)
    x = sequence.pad_sequences(x, maxlen=maxlen)
    return x


def generate_x_bert(file_path, tokenizer, maxlen):
    x = data_utils.read_features(file_path)
    x_seg = []
    temp = []
    for sample in x:
        x_seg.append([0] * maxlen)
        sample = re.sub('\s', '', sample)
        sample_char = []
        sample_char.append('[CLS]')
        for c in sample:
            sample_char.append(c)
        sample_char.append('[SEP]')
        temp.append(' '.join(sample_char))
    x = temp
    x = tokenizer.texts_to_sequences(x)
    x = sequence.pad_sequences(x, maxlen=maxlen, padding='post')
    return x, np.asarray(x_seg)


def generate_y(file_path):
    y = data_utils.read_field(file_path, 2, separator=datasets.delimeter)
    y = [[int(label) for label in labels.split(' ')] for labels in y]
    y = np.array(y)
    return y
