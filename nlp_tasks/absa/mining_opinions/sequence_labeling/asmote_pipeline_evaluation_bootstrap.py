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
parser.add_argument('--current_dataset', help='dataset name', default='ASMOTEDataRest14', type=str)
parser.add_argument('--ate_result_filepath_template', help='ate result filepath',
                    default=os.path.join(common_path.project_dir, 'AGF-ASOTE-data', 'absa', 'ASMOTE-prediction-result', 'ATE', 'result_of_predicting_test.txt'), type=str)
parser.add_argument('--atsa_result_filepath_template', help='atsa result filepath',
                    default=os.path.join(common_path.project_dir, 'AGF-ASOTE-data', 'absa', 'ASMOTE-prediction-result', 'ATSA', 'result_of_predicting_test.txt'), type=str)
parser.add_argument('--towe_result_filepath_template', help='towe result filepath',
                    default=os.path.join(common_path.project_dir, 'AGF-ASOTE-data', 'absa', 'ASMOTE-prediction-result', 'TOWE', 'result_of_predicting_test.txt'), type=str)
parser.add_argument('--debug', help='debug', default=False, type=argument_utils.my_bool)
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
        result[sentence] = [aspect_term_str_to_dict(aspect_term) for aspect_term in aspect_terms]
    return result


def read_atsa_result(filepath):
    """

    :param filepath:
    :return:
    """
    result = defaultdict(list)
    line_dicts = generate_line_dicts(filepath)
    for line_dict in line_dicts:
        sentence = line_dict['text']
        aspect_term = aspect_term_str_to_dict(line_dict['aspect_term'])
        polarity = line_dict['sentiment']
        result[sentence].append({'aspect_term': aspect_term, 'polarity': polarity})
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


def merge_results_of_subtasks(ate_result, atsa_result, towe_result):
    """

    :param ate_result:
    :param atsa_result:
    :param towe_result:
    :return:
    """
    result = {}
    for sentence in ate_result.keys():
        aspect_terms = ate_result[sentence]
        polarities = atsa_result[sentence]
        opinions = towe_result[sentence]
        result[sentence] = {'aspect_terms': aspect_terms, 'polarities': polarities, 'opinions': opinions}
    return result


def generate_subtasks_true(test_data):
    """

    :param test_data:
    :return:
    """
    ate_true = defaultdict(list)
    atsa_ture = defaultdict(list)
    towe_true = defaultdict(list)
    for sample in test_data:
        original_line_data = sample.metadata['original_line_data']
        sentence = original_line_data['sentence']
        aspect_term = original_line_data['aspect_term']
        opinions = original_line_data['opinions']
        polarity = original_line_data['polarity']

        ate_true[sentence].append(aspect_term)
        atsa_ture[sentence].append({'aspect_term': aspect_term, 'polarity': polarity})
        towe_true[sentence].append({'aspect_term': aspect_term, 'opinions': opinions})
    return ate_true, atsa_ture, towe_true


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


def evaluate_asmote(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    true_triplet_num = 0
    pred_triplet_num = 0
    tp = 0
    for sentence in sentences_true.keys():
        sentence_true = sentences_true[sentence]
        sentence_pred = sentences_pred[sentence]

        triplets_true = triplets_of_sentence(sentence_true)
        triplets_pred = triplets_of_sentence(sentence_pred)

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


def evaluate_towe(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    true_aspect_opinion_num = 0
    pred_aspect_opinion_num = 0
    tp = 0
    for sentence in sentences_true.keys():
        sentence_true = sentences_true[sentence]
        sentence_pred = sentences_pred[sentence]

        aspect_opinions_true = aspect_opinions_of_sentence(sentence_true)
        aspect_opinions_pred = aspect_opinions_of_sentence(sentence_pred)

        true_aspect_opinion_num += len(aspect_opinions_true)
        pred_aspect_opinion_num += len(aspect_opinions_pred)

        for e in aspect_opinions_true:
            if e in aspect_opinions_pred:
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


test_data = train_dev_test_data['test']
ate_true, asta_true, towe_true = generate_subtasks_true(test_data)
sentences_true = merge_results_of_subtasks(ate_true, asta_true, towe_true)

ate_result_filepath_template = configuration['ate_result_filepath_template']
atsa_result_filepath_template = configuration['atsa_result_filepath_template']
towe_result_filepath_template = configuration['towe_result_filepath_template']

run_num = 5
asmote_metrics_of_multi_runs = []
ate_metrics_of_multi_runs = []
atsa_metrics_of_multi_runs = []
towe_metrics_of_multi_runs = []
for i in range(run_num):
    if args.debug:
        ate_result_filepath = ate_result_filepath_template
        atsa_result_filepath = atsa_result_filepath_template
        towe_result_filepath = towe_result_filepath_template
    else:
        ate_result_filepath = ate_result_filepath_template % i
        atsa_result_filepath = atsa_result_filepath_template % i
        towe_result_filepath = towe_result_filepath_template % i
    if not os.path.exists(ate_result_filepath):
        print('not exist: %s' % ate_result_filepath)
        continue

    if not os.path.exists(atsa_result_filepath):
        print('not exist: %s' % atsa_result_filepath)
        continue

    if not os.path.exists(towe_result_filepath):
        print('not exist: %s' % towe_result_filepath)
        continue

    ate_pred = read_ate_result(ate_result_filepath)
    atsa_pred = read_atsa_result(atsa_result_filepath)
    towe_pred = read_towe_result(towe_result_filepath)
    sentences_pred = merge_results_of_subtasks(ate_pred, atsa_pred, towe_pred)

    asmote_metrics_of_multi_runs.append(evaluate_asmote(sentences_true, sentences_pred))
    ate_metrics_of_multi_runs.append(evaluate_ate(sentences_true, sentences_pred))
    atsa_metrics_of_multi_runs.append(evaluate_atsa(sentences_true, sentences_pred))
    towe_metrics_of_multi_runs.append(evaluate_towe(sentences_true, sentences_pred))


print_precision_recall_f1(asmote_metrics_of_multi_runs, 'asmote_metrics_of_multi_runs')
print()
print_precision_recall_f1(towe_metrics_of_multi_runs, 'towe_metrics_of_multi_runs')
print()
print_acc(atsa_metrics_of_multi_runs, 'atsa_metrics_of_multi_runs')
print()
print_precision_recall_f1(ate_metrics_of_multi_runs, 'ate_metrics_of_multi_runs')





