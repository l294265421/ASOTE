# -*- coding: utf-8 -*-


import argparse
import sys
import random
import copy
from typing import List
import json
from collections import defaultdict
import os

import torch
import numpy

from nlp_tasks.absa.utils import argument_utils
from nlp_tasks.absa.mining_opinions.data_adapter import data_object
from nlp_tasks.utils import file_utils
from nlp_tasks.common import common_path

parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='ASOTEDataRest14', type=str)
parser.add_argument('--version', help='dataset version', default='v2', type=str)
parser.add_argument('--ate_result_filepath_template', help='ate result filepath',
                    default=os.path.join(common_path.project_dir, 'AGF-ASOTE-data', 'absa', 'ASOTE-prediction-result', 'ATE', 'result_of_predicting_test.txt'), type=str)
parser.add_argument('--so_result_filepath_template', help='triplet result filepath',
                    default=os.path.join(common_path.project_dir, 'AGF-ASOTE-data', 'absa', 'ASOTE-prediction-result', 'unified_tag', 'result_of_predicting_test.txt.add_predicted_aspect_term'), type=str)
parser.add_argument('--debug', help='debug', default=False, type=argument_utils.my_bool)
args = parser.parse_args()

configuration = args.__dict__

dataset_name = configuration['current_dataset']
dataset = data_object.get_dataset_class_by_name(dataset_name)(configuration)
train_dev_test_data = dataset.get_data_type_and_data_dict()


def generate_line_dicts(filepath: str):
    """

    :param filepath:
    :return:
    """
    lines = file_utils.read_all_lines(filepath)
    for line in lines:
        line_dict = json.loads(line)
        yield line_dict


def aspect_term_str_to_dict(aspect_term_str: str):
    """

    :param aspect_term_str:
    :return:
    """
    parts = aspect_term_str.split('-')
    aspect_term_text = '-'.join(parts[:-2])
    start = int(parts[-2])
    end = int(parts[-1])
    result = {'start': start, 'end': end, 'term': aspect_term_text}
    return result


def aspect_term_dict_to_str(aspect_term_dict: dict):
    """

    :param aspect_term_dict:
    :return:
    """
    result = '%s-%d-%d' % (aspect_term_dict['term'], aspect_term_dict['start'], aspect_term_dict['end'])
    return result


def read_ate_result(filepath):
    """

    :param filepath:
    :return:
    """
    result = {}
    line_dicts = generate_line_dicts(filepath)
    for line_dict in line_dicts:
        sentence = line_dict['text']
        aspect_terms = line_dict['pred']
        result[sentence] = aspect_terms
    return result


def read_tosc_result(filepath):
    """

    :param filepath:
    :return:
    """
    result = {}
    line_dicts = generate_line_dicts(filepath)
    for line_dict in line_dicts:
        sentence = line_dict['text']
        if sentence not in result:
            result[sentence] = []
        aspect_term = aspect_term_dict_to_str(line_dict['opinion']['aspect_term'])
        opinion_term = aspect_term_dict_to_str(line_dict['opinion']['opinion_term'])
        polarity = line_dict['sentiment_pred']
        result[sentence].append({'aspect_term': aspect_term, 'opinion_term': opinion_term, 'polarity': polarity})
    return result


def read_towe_result(filepath):
    """

    :param filepath:
    :return:
    """
    result = defaultdict(list)
    line_dicts = generate_line_dicts(filepath)
    for line_dict in line_dicts:
        sentence = line_dict['text']
        aspect_term = aspect_term_str_to_dict(line_dict['aspect_terms'][0])
        opinion_terms = [aspect_term_str_to_dict(e) for e in line_dict['pred']]
        result[sentence].append({'aspect_term': aspect_term, 'opinions': opinion_terms})
    return result


def merge_results_of_subtasks(ate_result, tosc_result, towe_result):
    """

    :param ate_result:
    :param tosc_result:
    :param towe_result:
    :return:
    """
    result = {}
    for sentence in ate_result.keys():
        if sentence not in result:
            result[sentence] = []

        aspect_terms = ate_result[sentence]

        opinions = towe_result[sentence]
        aspect_and_opinions = {}
        for opinion in opinions:
            aspect_term = aspect_term_dict_to_str(opinion['aspect_term'])
            opinion_terms = [aspect_term_dict_to_str(e) for e in opinion['opinions']]
            aspect_and_opinions[aspect_term] = opinion_terms

        aspect_opinion_and_sentiment = {}
        if sentence in tosc_result:
            polarities = tosc_result[sentence]
            for polarity in polarities:
                aspect_opinion_and_sentiment['%s_%s' % (polarity['aspect_term'], polarity['opinion_term'])] = \
                    polarity['polarity']

        for aspect in aspect_terms:
            if aspect not in aspect_and_opinions or len(aspect_and_opinions[aspect]) == 0:
                continue
            opinions = aspect_and_opinions[aspect]
            for opinion in opinions:
                key = '%s_%s' % (aspect, opinion)
                sentiment = '-'
                if key in aspect_opinion_and_sentiment:
                    sentiment = aspect_opinion_and_sentiment[key]
                result[sentence].append('%s_%s_%s' % (aspect, sentiment, opinion))
    return result


def generate_subtasks_true(test_data):
    """

    :param test_data:
    :return:
    """
    sentence_and_triplets = {}
    triplet_num_true = 0
    for sample in test_data:
        original_line_data = sample.metadata['original_line_data']
        sentence = original_line_data['sentence']
        if sentence not in sentence_and_triplets:
            sentence_and_triplets[sentence] = []

        opinions = original_line_data['opinions']
        for opinion in opinions:
            if 'polarity' not in opinion:
                continue
            aspect_term = opinion['aspect_term']
            opinion_term = opinion['opinion_term']
            triplet_str = '%s_%s_%s' % (aspect_term_dict_to_str(aspect_term), opinion['polarity'],
                                        aspect_term_dict_to_str(opinion_term))
            sentence_and_triplets[sentence].append(triplet_str)
            triplet_num_true += 1
    print('triplet_num_true: %d' % triplet_num_true)
    return sentence_and_triplets


def triplets_of_sentence(results_of_subtasks_of_sentence):
    """

    :param results_of_subtasks_of_sentence:
    :return:
    """
    aspect_terms = results_of_subtasks_of_sentence['aspect_terms']
    polarities = results_of_subtasks_of_sentence['polarities']
    opinions = results_of_subtasks_of_sentence['opinions']

    aspect_term_strs = [aspect_term_dict_to_str(e) for e in aspect_terms]

    aspect_term_str_and_sentiment = {}
    for e in polarities:
        aspect_term_str_and_sentiment[aspect_term_dict_to_str(e['aspect_term'])] = e['polarity']

    aspect_term_str_and_opinions = {}
    for e in opinions:
        aspect_term_str = aspect_term_dict_to_str(e['aspect_term'])
        opinions_of_this_aspect: List = e['opinions']
        if len(opinions_of_this_aspect) == 0:
            continue
        opinions_of_this_aspect.sort(key=lambda x: x['start'])
        opinion_strs_of_this_aspect = [aspect_term_dict_to_str(e) for e in opinions_of_this_aspect]
        aspect_term_str_and_opinions[aspect_term_str] = '_'.join(opinion_strs_of_this_aspect)
    results = set()
    for aspect_term_str in aspect_term_strs:
        sentiment = '-'
        # aspect term，，ground truth aspect term，-
        # (，aspect term)
        if aspect_term_str in aspect_term_str_and_sentiment:
            sentiment = aspect_term_str_and_sentiment[aspect_term_str]

        if aspect_term_str in aspect_term_str_and_opinions:
            multiple_opinions = aspect_term_str_and_opinions[aspect_term_str]
            results.add('%s__%s__%s' % (aspect_term_str, sentiment, multiple_opinions))
    return results


def aspect_terms_of_sentence(results_of_subtasks_of_sentence):
    """

    :param results_of_subtasks_of_sentence:
    :return:
    """
    aspect_terms = results_of_subtasks_of_sentence['aspect_terms']

    result = set([aspect_term_dict_to_str(e) for e in aspect_terms])
    return result


def aspect_sentiment_of_sentence(results_of_subtasks_of_sentence):
    """

    :param results_of_subtasks_of_sentence:
    :return:
    """
    polarities = results_of_subtasks_of_sentence['polarities']

    result = set()
    for e in polarities:
        result.add('%s__%s' % (aspect_term_dict_to_str(e['aspect_term']), e['polarity']))
    return result


def aspect_opinions_of_sentence(results_of_subtasks_of_sentence):
    """

    :param results_of_subtasks_of_sentence:
    :return:
    """
    opinions = results_of_subtasks_of_sentence['opinions']

    result = set()
    for e in opinions:
        aspect_term_str = aspect_term_dict_to_str(e['aspect_term'])
        opinions_of_this_aspect: List = e['opinions']
        if len(opinions_of_this_aspect) == 0:
            continue
        opinions_of_this_aspect.sort(key=lambda x: x['start'])
        opinion_strs_of_this_aspect = [aspect_term_dict_to_str(e) for e in opinions_of_this_aspect]
        for opinion_str_of_this_aspect in opinion_strs_of_this_aspect:
            result.add('%s__%s' % (aspect_term_str, opinion_str_of_this_aspect))
    return result


def get_metrics(true_num, pred_num, tp):
    """

    :param true_num:
    :param pred_num:
    :param tp:
    :return:
    """
    precision = tp / pred_num
    recall = tp / true_num
    f1 = 2 * precision * recall / (precision + recall)
    return {'precision': '%.3f' % (precision * 100), 'recall': '%.3f' % (recall * 100), 'f1': '%.3f' % (f1 * 100)}


def evaluate_asote(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    true_triplet_num = 0
    pred_triplet_num = 0
    tp = 0
    for sentence in sentences_pred.keys():
        triplets_true = sentences_true[sentence]
        triplets_pred = sentences_pred[sentence]

        true_triplet_num += len(triplets_true)
        pred_triplet_num += len(triplets_pred)

        for e in triplets_true:
            if e in triplets_pred:
                tp += 1
    result = get_metrics(true_triplet_num, pred_triplet_num, tp)
    return result


def remove_sentiment(triplets: List[str]):
    result = []
    for e in triplets:
        e = e.replace('_positive_', '_')
        e = e.replace('_negative_', '_')
        e = e.replace('_neutral_', '_')
        e = e.replace('_-_', '_')
        result.append(e)
    return result


def evaluate_ao_pair(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    true_triplet_num = 0
    pred_triplet_num = 0
    tp = 0
    for sentence in sentences_pred.keys():
        triplets_true_temp = sentences_true[sentence]
        triplets_pred_temp = sentences_pred[sentence]

        triplets_true = remove_sentiment(triplets_true_temp)
        triplets_pred = remove_sentiment(triplets_pred_temp)

        true_triplet_num += len(triplets_true)
        pred_triplet_num += len(triplets_pred)

        for e in triplets_true:
            if e in triplets_pred:
                tp += 1
    result = get_metrics(true_triplet_num, pred_triplet_num, tp)
    return result


def evaluate_ate(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    true_aspect_term_num = 0
    pred_aspect_term_num = 0
    tp = 0
    for sentence in sentences_true.keys():
        sentence_true = sentences_true[sentence]
        sentence_pred = sentences_pred[sentence]

        aspect_terms_true = aspect_terms_of_sentence(sentence_true)
        aspect_terms_pred = aspect_terms_of_sentence(sentence_pred)

        true_aspect_term_num += len(aspect_terms_true)
        pred_aspect_term_num += len(aspect_terms_pred)

        for e in aspect_terms_true:
            if e in aspect_terms_pred:
                tp += 1
    result = get_metrics(true_aspect_term_num, pred_aspect_term_num, tp)
    return result


def evaluate_atsa(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    total_num = 0
    correct_num = 0
    for sentence in sentences_true.keys():
        sentence_true = sentences_true[sentence]
        sentence_pred = sentences_pred[sentence]

        aspect_sentiment_true = aspect_sentiment_of_sentence(sentence_true)
        aspect_sentiment_pred = aspect_sentiment_of_sentence(sentence_pred)

        total_num += len(aspect_sentiment_true)

        for e in aspect_sentiment_true:
            if e in aspect_sentiment_pred:
                correct_num += 1
    result = {'accuracy': '%.3f' % (correct_num / total_num * 100)}
    return result


def sentence_triplets_to_sentence_aspect_and_opinions(sentence_triplets):
    """

    :param sentence_triplets:
    :return:
    """
    result = defaultdict(list)
    for sentence, triplets in sentence_triplets.items():
        for triplet in triplets:
            delimeter = '-'
            if '_positive_' in triplet:
                delimeter = '_positive_'
            elif '_negative_' in triplet:
                delimeter = '_negative_'
            elif '_neutral_' in triplet:
                delimeter = '_neutral_'
            parts = triplet.split(delimeter)
            key = '%s_%s' % (sentence, parts[0])
            result[key].append(parts[1])
    return result


def evaluate_towe(so_pred):
    """

    :param so_pred:
    :return:
    """
    sentences_true = {}
    sentences_pred = {}
    for e in so_pred:
        e = json.loads(e)
        sentence = ' '.join(e['words'])

        aspect = aspect_term_dict_to_str(e['word_indices_of_aspect_terms'])
        key = '%s_%s' % (sentence, aspect)
        if key not in sentences_pred:
            sentences_pred[key] = []
            sentences_true[key] = []

        for opinion in e['opinions']:
            sentences_pred[key].append(aspect_term_dict_to_str(opinion))
        for opinion in e['opinions_true']:
            sentences_true[key].append(aspect_term_dict_to_str(opinion))

    true_aspect_opinion_num = 0
    pred_aspect_opinion_num = 0
    tp = 0
    for sentence in sentences_true.keys():
        triplets_true = sentences_true[sentence]
        triplets_pred = sentences_pred[sentence]

        true_aspect_opinion_num += len(triplets_true)
        pred_aspect_opinion_num += len(triplets_pred)

        for e in triplets_true:
            if e in triplets_pred:
                tp += 1
    result = get_metrics(true_aspect_opinion_num, pred_aspect_opinion_num, tp)
    return result


def print_precision_recall_f1(metrics_of_multi_runs, description: str = ''):
    """

    :param metrics_of_multi_runs:
    :param description:
    :return:
    """
    print(description)
    precisions = []
    recalls = []
    f1s = []
    for e in metrics_of_multi_runs:
        precisions.append(e['precision'])
        recalls.append(e['recall'])
        f1s.append(e['f1'])
    print('precision: %s' % ','.join(precisions))
    print('recall: %s' % ','.join(recalls))
    print('f1: %s' % ','.join(f1s))
    print('%s\t%s\t%s' % (','.join(precisions), ','.join(recalls), ','.join(f1s)))


def print_acc(metrics_of_multi_runs, description: str = ''):
    """

    :param metrics_of_multi_runs:
    :param description:
    :return:
    """
    print(description)
    print('acc: %s' % ','.join([e['accuracy'] for e in metrics_of_multi_runs]))


def get_sentence_and_triplets_pred(so_pred, ate_pred):
    """
    :param triplets_pred:
    :param ate_pred:
    :return:
    """
    result = {}
    triplet_num_pred = 0
    for e in so_pred:
        e = json.loads(e)
        sentence = ' '.join(e['words'])
        if sentence not in result:
            result[sentence] = []

        aspect = aspect_term_dict_to_str(e['word_indices_of_aspect_terms'])
        aspects_pred = ate_pred[sentence]
        if aspect not in aspects_pred:
            continue
        pred = e['opinions']
        for opinion in pred:
            triplet_num_pred += 1
            opinion_str = '%s_%s_%s' % (aspect,
                                        opinion['polarity'],
                                        aspect_term_dict_to_str(opinion))
            result[sentence].append(opinion_str)
    print('triplet_num_pred: %d' % triplet_num_pred)
    return result


test_data = train_dev_test_data['test']
triplets_true = generate_subtasks_true(test_data)

run_num = 5
asote_metrics_of_multi_runs = []
for i in range(run_num):
    if args.debug:
        ate_result_filepath = args.ate_result_filepath_template
        triplet_result_filepath = args.so_result_filepath_template
    else:
        ate_result_filepath = args.ate_result_filepath_template % i
        triplet_result_filepath = args.so_result_filepath_template % i

    if not os.path.exists(ate_result_filepath):
        print('not exist: %s' % ate_result_filepath)
        continue

    if not os.path.exists(triplet_result_filepath):
        print('not exist: %s' % triplet_result_filepath)
        continue

    ate_pred = read_ate_result(ate_result_filepath)
    so_pred = file_utils.read_all_lines(triplet_result_filepath)

    triplets_pred = get_sentence_and_triplets_pred(so_pred, ate_pred)

    asote_metrics_of_multi_runs.append(evaluate_asote(triplets_true, triplets_pred))


print_precision_recall_f1(asote_metrics_of_multi_runs, 'asote_metrics_of_multi_runs')

print('-' * 100)
asote_metrics_of_multi_runs = []
for i in range(run_num):
    if args.debug:
        ate_result_filepath = args.ate_result_filepath_template
        triplet_result_filepath = args.so_result_filepath_template
    else:
        ate_result_filepath = args.ate_result_filepath_template % i
        triplet_result_filepath = args.so_result_filepath_template % i

    if not os.path.exists(ate_result_filepath):
        print('not exist: %s' % ate_result_filepath)
        continue

    if not os.path.exists(triplet_result_filepath):
        print('not exist: %s' % triplet_result_filepath)
        continue

    ate_pred = read_ate_result(ate_result_filepath)
    so_pred = file_utils.read_all_lines(triplet_result_filepath)

    triplets_pred = get_sentence_and_triplets_pred(so_pred, ate_pred)

    asote_metrics_of_multi_runs.append(evaluate_ao_pair(triplets_true, triplets_pred))


print_precision_recall_f1(asote_metrics_of_multi_runs, 'ao_pair_metrics_of_multi_runs')

print('-' * 100)
asote_metrics_of_multi_runs = []
for i in range(run_num):
    if args.debug:
        triplet_result_filepath = args.so_result_filepath_template[: -1 * len('.add_predicted_aspect_term')]
    else:
        triplet_result_filepath = args.so_result_filepath_template[: -1 * len('.add_predicted_aspect_term')] % i

    if not os.path.exists(triplet_result_filepath):
        print('not exist: %s' % triplet_result_filepath)
        continue

    so_pred = file_utils.read_all_lines(triplet_result_filepath)

    asote_metrics_of_multi_runs.append(evaluate_towe(so_pred))


print_precision_recall_f1(asote_metrics_of_multi_runs, 'towe_metrics_of_multi_runs')
