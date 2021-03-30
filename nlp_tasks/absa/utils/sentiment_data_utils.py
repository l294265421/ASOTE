import numpy as np
from keras.preprocessing import sequence

from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.utils import evaluate_utils
from nlp_tasks.absa.conf import datasets, task_conf


def generate_x(file_path, tokenizer, maxlen):
    x = data_utils.read_field(file_path, 1, separator=task_conf.delimeter)
    x = tokenizer.texts_to_sequences(x)
    x = sequence.pad_sequences(x, maxlen=maxlen)
    return x


def generate_x_subject(file_path, tokenizer, maxlen, repeat_subject=True):
    x = data_utils.read_field(file_path, 1, separator=datasets.delimeter)
    x = tokenizer.texts_to_sequences(x)
    x_len = [len(element) for element in x]

    x_subjects = data_utils.read_field(file_path, 2, separator=datasets.delimeter)
    x_subjects = [e for e in x_subjects]
    if repeat_subject:
        x_subjects = data_utils.repeat_element_in_list(x_subjects, x_len)
    x_subjects = tokenizer.texts_to_sequences(x_subjects)
    x_subjects = sequence.pad_sequences(x_subjects, maxlen=1)
    if repeat_subject:
        x_subjects = sequence.pad_sequences(x_subjects, maxlen=maxlen)
    else:
        x_subjects = np.array(x_subjects)
    return x_subjects


# def generate_x_subject_joint_model(file_path, tokenizer, maxlen, repeat_subject=True):
#     x = data_utils.read_field(file_path, 1, separator=task_conf.delimeter)
#     x = tokenizer.texts_to_sequences(x)
#     x_len = [len(element) for element in x]
#     result = []
#     for i in range(len(label_mapping.subject_mapping_reverse)):
#         subject = label_mapping.subject_mapping_reverse[str(i)]
#         x_subjects = [subject] * len(x)
#         if repeat_subject:
#             x_subjects = data_utils.repeat_element_in_list(x_subjects, x_len)
#         x_subjects = tokenizer.texts_to_sequences(x_subjects)
#         if repeat_subject:
#             x_subjects = sequence.pad_sequences(x_subjects, maxlen=maxlen)
#         else:
#             x_subjects = np.array(x_subjects)
#         result.append(x_subjects)
#     return result


def generate_x_subject_joint_model(sample_num, subject_class_num):
    result = []
    for i in range(subject_class_num):
        one_subject = []
        for j in range(sample_num):
            one_subject.append([i + 1])
        result.append(np.array(one_subject))
    return result


def generate_x_punctuation(file_path, tokenizer, maxlen):
    x = data_utils.read_field(file_path, 1, separator=datasets.delimeter)

    x_punctuation = []
    punctuations = '，。？！；…'
    for sample in x:
        words = sample.split()
        x_punctuation_sample = []
        punctuation = '。'
        for i in range(-1, len(words) * -1 - 1, -1):
            if words[i] in punctuations:
                punctuation = words[i]
            x_punctuation_sample.append(punctuation)
        x_punctuation_sample.reverse()
        x_punctuation.append(' '.join(x_punctuation_sample))

    x_punctuation = tokenizer.texts_to_sequences(x_punctuation)
    x_punctuation = sequence.pad_sequences(x_punctuation, maxlen=maxlen)
    return x_punctuation


def generate_y(file_path):
    y = data_utils.read_field(file_path, 3, separator=datasets.delimeter)
    y = [[int(label) for label in labels.split(' ')] for labels in y]
    y = np.array(y)
    return y


def generate_y_joint_model(file_path):
    y_aspect = data_utils.read_field(file_path, 2, separator=task_conf.delimeter)
    y_aspect = [[int(label) for label in labels.split(' ')] for labels in y_aspect]
    y_aspect = np.array(y_aspect)
    y_aspect = evaluate_utils.to_multi_output_label(y_aspect)

    y = []
    y.extend(y_aspect)

    for i in range(len(label_mapping.subject_mapping_reverse)):
        y_sentiment = data_utils.read_field(file_path, 3 + i, separator=task_conf.delimeter)
        y_sentiment = [[int(label) for label in labels.split(' ')] for labels in y_sentiment]
        y_sentiment = np.array(y_sentiment)
        y.append(y_sentiment)
    return y
