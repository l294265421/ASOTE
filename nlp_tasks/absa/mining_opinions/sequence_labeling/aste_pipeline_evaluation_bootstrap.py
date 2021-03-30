# -*- coding: utf-8 -*-


import argparse
import sys
import random
import copy
from typing import List
import json
from collections import defaultdict

import torch
import numpy

from nlp_tasks.absa.utils import argument_utils
from nlp_tasks.absa.mining_opinions.data_adapter import data_object
from nlp_tasks.utils import file_utils

parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='triplet_rest14', type=str)
parser.add_argument('--ate_result_filepath_template', help='ate result filepath',
                    default=r'', type=str)
parser.add_argument('--atsa_result_filepath_template', help='atsa result filepath',
                    default=r'', type=str)
parser.add_argument('--towe_result_filepath_template', help='towe result filepath',
                    default=r'', type=str)
args = parser.parse_args()

configuration = args.__dict__


def first_term_from_tags(tags: List[str], start_index: int):
    """

    :param tags:
    :param start_index:
    :return:
    """
    if 'B' in tags[start_index:]:
        start_index = tags.index('B', start_index)
        end_index = start_index + 1
        while end_index < len(tags) and tags[end_index] == 'I':
            end_index += 1
        return [start_index, end_index]
    else:
        return None


def terms_from_tags(tags: List[str], words: List[str]):
    """

    :param tags:
    :return:
    """
    tags = tags[: len(words)]

    terms = []
    start_index = 0
    while start_index < len(tags):
        term = first_term_from_tags(tags, start_index)
        if term is None:
            break
        else:
            terms.append(term)
            start_index = term[1]

    term_with_texts = []
    for term in terms:
        term_text = ' '.join(words[term[0]: term[1]])
        term_with_texts.append('%s-%d-%d' % (term_text, term[0], term[1]))
    return term_with_texts


def precision_recall_f1(pred: set, true: set):
    """

    :param pred:
    :param true:
    :return:
    """
    intersection = pred.intersection(true)
    precision = len(intersection) / len(pred)
    recall = len(intersection) / len(true)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def print_precision_recall_f1(metrics: dict):
    """

    :param metrics:
    :return:
    """
    precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
    result = '\t'.join([','.join(precision), ','.join(recall), ','.join(f1)])
    print(result)


dataset_name = configuration['current_dataset']
dataset = data_object.get_dataset_class_by_name(dataset_name)()
train_dev_test_data = dataset.get_data_type_and_data_dict()

data_type_and_max_len = {}
for data_type, data in train_dev_test_data.items():
    max_sentence_len = 0
    for sample in data:
        sentence: data_object.Sentence = sample
        text = sentence.text
        words = sentence.words
        if len(words) > max_sentence_len:
            max_sentence_len = len(words)
    data_type_and_max_len[data_type] = max_sentence_len

print('max_sentence_len: %s' % str(data_type_and_max_len))

golden_test_data = train_dev_test_data['test']
golden_aspect_opinion_sentiment_triplets = set()
golden_aspect_opinion_pairs = set()
golden_aspect_sentiment_pairs = set()

polarity_mapping = {
    'POS': 'positive',
    'NEG': 'negative',
    'NEU': 'neutral'
}

for sample in golden_test_data:
    sentence: data_object.Sentence = sample
    text = sentence.text
    words = sentence.words
    sentiment = polarity_mapping[sentence.polarity]

    target_tags = sentence.target_tags
    aspect_term = terms_from_tags(target_tags, words)[0]

    golden_aspect_sentiment_pairs.add('%s-%s-%s' % (text, aspect_term, sentiment))

    opinion_tags = sentence.opinion_words_tags
    opinion_terms = terms_from_tags(opinion_tags, words)
    if len(opinion_terms) == 0:
        opinion_terms = ['-']
    for opinion_term in opinion_terms:
        golden_aspect_opinion_pairs.add('%s-%s-%s' % (text, aspect_term, opinion_term))
        golden_aspect_opinion_sentiment_triplets.add('%s-%s-%s-%s' % (text, aspect_term, opinion_term, sentiment))


aspect_sentiment_pair_metrics_all = defaultdict(list)
aspect_opinion_pair_metrics_all = defaultdict(list)
aste_metrics_all = defaultdict(list)

for i in range(5):
    ate_result_filepath = configuration['ate_result_filepath_template'] % i
    # print('ate_result_filepath = "%s"' % ate_result_filepath)
    ate_lines = file_utils.read_all_lines(ate_result_filepath)
    pred_text_aspects = set()
    for line in ate_lines:
        line_dict = json.loads(line)
        for aspect_term in line_dict['pred']:
            text_aspect = '%s-%s' % (line_dict['text'], aspect_term)
            pred_text_aspects.add(text_aspect)

    atsa_result_filepath = configuration['atsa_result_filepath_template'] % i
    # print('atsa_result_filepath = "%s"' % atsa_result_filepath)
    atsa_lines = file_utils.read_all_lines(atsa_result_filepath)
    pred_text_aspect_sentiment = {}

    for line in atsa_lines:
        line_dict = json.loads(line)
        text = line_dict['text']
        aspect = line_dict['aspect_term']
        sentiment = line_dict['sentiment']
        text_aspect = '%s-%s' % (text, aspect)
        pred_text_aspect_sentiment[text_aspect] = sentiment

    towe_result_filepath = configuration['towe_result_filepath_template'] % i
    # print('towe_result_filepath = "%s"' % towe_result_filepath)
    towe_lines = file_utils.read_all_lines(towe_result_filepath)
    pred_text_aspect_opinions = {}
    for line in towe_lines:
        line_dict = json.loads(line)
        text = line_dict['text']
        aspect = line_dict['aspect_terms'][0]
        opinions = line_dict['pred']
        text_aspect = '%s-%s' % (text, aspect)
        pred_text_aspect_opinions[text_aspect] = opinions

    # case analysis
    # 正确预测了aspect term，但是没有在标准数据集，我们需要扩展数据集，使之包含没有观点词的数据
    # 'While there \'s a decent menu , it should n\'t take ten minutes to get your drinks and 45 for a dessert pizza .-dessert pizza-21-23'
    pred_aspect_sentiment_pairs = set()
    pred_aspect_opinion_pairs = set()
    pred_text_aspect_opinion_sentiment = set()
    for text_aspect in pred_text_aspects:
        sentiment = '-'
        if text_aspect in pred_text_aspect_sentiment:
            sentiment = pred_text_aspect_sentiment[text_aspect]
        pred_aspect_sentiment_pairs.add('%s-%s' % (text_aspect, sentiment))
        opinions = ['-']
        if text_aspect in pred_text_aspect_opinions and len(pred_text_aspect_opinions[text_aspect]) != 0:
            opinions = pred_text_aspect_opinions[text_aspect]
        for opinion in opinions:
            item = '%s-%s-%s' % (text_aspect, opinion, sentiment)
            pred_text_aspect_opinion_sentiment.add(item)
            pred_aspect_opinion_pairs.add('%s-%s' % (text_aspect, opinion))

    aspect_sentiment_pair_metrics = precision_recall_f1(pred_aspect_sentiment_pairs, golden_aspect_sentiment_pairs)
    for key, value in aspect_sentiment_pair_metrics.items():
        aspect_sentiment_pair_metrics_all[key].append('%.5f' % value)

    aspect_opinion_pair_metrics = precision_recall_f1(pred_aspect_opinion_pairs, golden_aspect_opinion_pairs)
    for key, value in aspect_opinion_pair_metrics.items():
        aspect_opinion_pair_metrics_all[key].append('%.5f' % value)

    aste_metrics = precision_recall_f1(pred_text_aspect_opinion_sentiment, golden_aspect_opinion_sentiment_triplets)
    for key, value in aste_metrics.items():
        aste_metrics_all[key].append('%.5f' % value)

print('aste_metrics_all:')
print_precision_recall_f1(aste_metrics_all)
print('aspect_opinion_pair_metrics:')
print_precision_recall_f1(aspect_opinion_pair_metrics_all)
print('aspect_sentiment_pair_metrics:')
print_precision_recall_f1(aspect_sentiment_pair_metrics_all)

# (0.6425287356321839, 0.5675126903553299, 0.6026954177897574)