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

parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='triplet_rest16_supporting_shared_opinion', type=str)
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
        term_with_texts.append((term_text, term[0], term[1]))
    return term_with_texts


dataset_name = configuration['current_dataset']
dataset = data_object.get_dataset_class_by_name(dataset_name)()
train_dev_test_data = dataset.get_data_type_and_data_dict()

data_type_and_samples = {}
for data_type, data in train_dev_test_data.items():
    samples = []
    for sample in data:
        sentence: data_object.Sentence = sample
        text = sentence.text
        words = sentence.words

        sentiment = sentence.polarity

        target_tags = sentence.target_tags
        aspect_term = terms_from_tags(target_tags, words)[0]

        opinion_words_tags = sentence.opinion_words_tags
        opinion_terms = terms_from_tags(opinion_words_tags, words)
        triplet = {
            'sentence': text,
            'words': words,
            'aspect_term': aspect_term,
            'opinion_terms': opinion_terms
        }
        samples.append(json.dumps(triplet, ensure_ascii=False))
    print('%s: %d' % (data_type, len(samples)))