# -*- coding: utf-8 -*-


import logging
import os
import json
import re
import csv
import pickle
import sys
from collections import defaultdict
from typing import Dict, List
import copy

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

from nlp_tasks.common import common_path
from nlp_tasks.utils import file_utils
from nlp_tasks.utils import sequence_labeling_utils

logger = logging.getLogger(__name__)
base_data_dir = common_path.get_task_data_dir('absa', is_original=True)


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


class AspectTerm:
    """
    aspect term
    """

    def __init__(self, term, polarity, from_index, to_index, category=None, metadata=None):
        self.term = term
        self.polarity = polarity
        # inclusive
        self.from_index = int(from_index)
        # exclusive
        self.to_index = int(to_index)
        self.category = category
        self.metadata = metadata

    def __str__(self):
        return '%s-%s-%s-%s-%s' % (self.term, str(self.polarity), str(self.from_index), str(self.to_index),
                                   self.category)


class Sentence:
    """
    """

    def __init__(self, text: str, words: List[str], target_tags: List[str],
                 opinion_words_tags, polarity='', metadata={}):
        self.text = text
        self.words = words
        self.target_tags = target_tags
        self.opinion_words_tags = opinion_words_tags
        self.polarity = polarity
        self.metadata = metadata

    def __str__(self):
        return ' '.join(['%s_%s_%s' % (self.words[i], self.target_tags[i], self.opinion_words_tags[i])
                         for i in range(len(self.words))])


class BaseDataset:
    """

    """

    def __init__(self, configuration: Dict):
        self.configuration = configuration
        self.train_data, self.dev_data, self.test_data = self._load_train_dev_test_data()

    def _load_train_dev_test_data(self):
        """
        加载数据
        :return:
        """
        return None, None, None

    def get_data_type_and_data_dict(self):
        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        return data_type_and_data


class TargetOrientedOpinionWordsExtraction(BaseDataset):
    """
    2019-naacl-Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling
    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        raise NotImplementedError()

    def _load_data_by_filepath(self, filepath):
        lines = file_utils.read_all_lines(filepath)
        lines = lines[1:]
        result = []
        for line in lines:
            parts = line.split('\t')
            text = parts[1]
            words = [tag.split('\\')[0] for tag in parts[2].split(' ')]
            target_tags = [tag.split('\\')[1] for tag in parts[2].split(' ')]
            opinion_words_tags = [tag.split('\\')[1] for tag in parts[3].split(' ')]
            sentence = Sentence(text, words, target_tags, opinion_words_tags)
            result.append(sentence)
        return result

    def _load_train_dev_test_data(self):
        base_dir = os.path.join(base_data_dir,
                                'TargetorientedOpinionWordsExtractionwithTargetfusedNeuralSequenceLabeling',
                                self._get_sub_dir())
        train_filepath = os.path.join(base_dir, 'train.tsv')
        test_filepath = os.path.join(base_dir, 'test.tsv')

        train_data = self._load_data_by_filepath(train_filepath)
        test_data = self._load_data_by_filepath(test_filepath)

        train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=1234)
        return train_data, dev_data, test_data


class Rest14(TargetOrientedOpinionWordsExtraction):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14res'


class Lapt14(TargetOrientedOpinionWordsExtraction):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14lap'


class Rest15(TargetOrientedOpinionWordsExtraction):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '15res'


class Rest16(TargetOrientedOpinionWordsExtraction):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '16res'


class ASOTEData(BaseDataset):
    """
    """

    def __init__(self, configuration: dict={}):
        super().__init__(configuration)

    def to_aspect_tags(self, aspect_term_dict: dict, words: List[str]):
        """
        aspect_term_dict only contains one aspect term
        :param aspect_term_dict:
        :param words:
        :return:
        """
        tags = ['O' for _ in words]
        start = aspect_term_dict['start']
        end = aspect_term_dict['end']
        tags[start] = 'B'
        for i in range(start + 1, end):
            tags[i] = 'I'
        return tags

    def to_opinion_tags(self, opinion_term_dicts: List[Dict], words: List[str]):
        """
        opinion_terms_dict may be contain zero, one, or multiple opinion terms
        :param opinion_term_dicts:
        :param words:
        :return:
        """
        tags = ['O' for _ in words]
        for opinion_term_dict in opinion_term_dicts:
            if 'opinion_term' not in opinion_term_dict:
                continue
            start = opinion_term_dict['opinion_term']['start']
            end = opinion_term_dict['opinion_term']['end']
            polarity = opinion_term_dict['polarity']
            tags[start] = '%s-B' % polarity
            for i in range(start + 1, end):
                tags[i] = '%s-I' % polarity
        return tags

    def to_opinion_tags_for_so(self, opinion_term_dicts: List[Dict], words: List[str]):
        """
        opinion_terms_dict may be contain zero, one, or multiple opinion terms
        :param opinion_term_dicts:
        :param words:
        :return:
        """
        tags = ['O' for _ in words]
        for opinion_term_dict in opinion_term_dicts:
            if 'opinion_term' not in opinion_term_dict:
                continue
            start = opinion_term_dict['opinion_term']['start']
            end = opinion_term_dict['opinion_term']['end']
            polarity = opinion_term_dict['polarity']
            tags[start] = '%s-B' % polarity
            for i in range(start + 1, end):
                tags[i] = '%s-I' % polarity
        return tags

    def to_opinion_tags_without_sentiment(self, opinion_term_dicts: List[Dict], words: List[str]):
        """
        opinion_terms_dict may be contain zero, one, or multiple opinion terms
        :param opinion_term_dicts:
        :param words:
        :return:
        """
        tags = ['O' for _ in words]
        for opinion_term_dict in opinion_term_dicts:
            if 'opinion_term' not in opinion_term_dict:
                continue
            start = opinion_term_dict['opinion_term']['start']
            end = opinion_term_dict['opinion_term']['end']
            tags[start] = 'B'
            for i in range(start + 1, end):
                tags[i] = 'I'
        return tags

    def _load_data_by_filepath(self, filepath):
        lines = file_utils.read_all_lines(filepath)
        result = []
        keys = set()
        for line in lines:
            original_line_data = json.loads(line)
            text = original_line_data['sentence']
            words = original_line_data['words']
            polarity = original_line_data['polarity']

            aspect_term_str = aspect_term_dict_to_str(original_line_data['aspect_term'])
            key = '%s_%s' % (text, aspect_term_str)
            keys.add(key)

            target_tags = self.to_aspect_tags(original_line_data['aspect_term'], words)

            opinion_words_tags = self.to_opinion_tags(original_line_data['opinions'], words)
            if self.configuration is not None and 'opinion_tag_with_sentiment' in self.configuration and not self.configuration['opinion_tag_with_sentiment']:
                opinion_words_tags = self.to_opinion_tags_without_sentiment(original_line_data['opinions'], words)

            opinion_words_tags_for_so = self.to_opinion_tags_for_so(original_line_data['opinions'], words)
            original_line_data['opinion_words_tags_for_so'] = opinion_words_tags_for_so

            metadata = {'original_line_data': original_line_data}
            sentence = Sentence(text, words, target_tags, opinion_words_tags, polarity=polarity,
                                metadata=metadata)
            result.append(sentence)

        if self.configuration is not None and 'add_predicted_aspect_term' in self.configuration and self.configuration['add_predicted_aspect_term']:
            ate_result_lines = file_utils.read_all_lines(self.configuration['ate_result_filepath'])
            for line in ate_result_lines:
                original_line_data = json.loads(line)

                text: str = original_line_data['text']
                words = text.split(' ')
                polarity = 'positive'

                for aspect_term_str in original_line_data['pred']:
                    key = '%s_%s' % (text, aspect_term_str)
                    if key in keys:
                        continue

                    aspect_term = aspect_term_str_to_dict(aspect_term_str)

                    target_tags = self.to_aspect_tags(aspect_term, words)

                    opinion_words_tags = self.to_opinion_tags([], words)

                    opinion_words_tags_for_so = self.to_opinion_tags_for_so([], words)
                    original_line_data['opinion_words_tags_for_so'] = opinion_words_tags_for_so

                    original_line_data_copy = copy.deepcopy(original_line_data)
                    original_line_data_copy['sentence'] = text
                    original_line_data_copy['words'] = copy.deepcopy(words)
                    original_line_data_copy['polarity'] = polarity
                    original_line_data_copy['aspect_term'] = aspect_term
                    original_line_data_copy['opinions'] = []

                    metadata = {'original_line_data': original_line_data_copy}
                    sentence = Sentence(text, original_line_data_copy['words'], target_tags, opinion_words_tags, polarity=polarity,
                                        metadata=metadata)
                    result.append(sentence)
        return result

    def _get_sub_dir(self):
        raise NotImplementedError()

    def _load_train_dev_test_data(self):
        base_dir = os.path.join(base_data_dir,
                                'ASOTE',
                                self._get_sub_dir(),
                                'asote_gold_standard'
                                )
        train_filepath = os.path.join(base_dir, 'train.txt')
        dev_filepath = os.path.join(base_dir, 'dev.txt')
        test_filepath = os.path.join(base_dir, 'test.txt')

        print('train data:')
        train_data = self._load_data_by_filepath(train_filepath)
        print('dev data:')
        dev_data = self._load_data_by_filepath(dev_filepath)
        print('test data:')
        test_data = self._load_data_by_filepath(test_filepath)

        return train_data, dev_data, test_data

    def generate_tosc_data(self, test_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for sentence in data:
                sentence: Sentence = sentence
                content = sentence.text
                label = []
                words = sentence.words
                for opinion in sentence.metadata['original_line_data']['opinions']:
                    if 'polarity' not in opinion:
                        continue
                    polarity = opinion['polarity']
                    distinct_polarities.add(polarity)

                    term = opinion['opinion_term']['term']
                    word_start = opinion['opinion_term']['start']
                    word_end = opinion['opinion_term']['end']

                    start = 0 if word_start == 0 else len(' '.join(words[: word_start])) + 1
                    end = len(' '.join(words[: word_end]))

                    term_temp = content[start: end]
                    if term_temp != term:
                        print('char index error, term: %s term_temp: %s' % (term, term_temp))

                    opinion_term = AspectTerm(term, polarity, start, end, metadata={'opinion': opinion})
                    label.append(opinion_term)
                if len(label) == 0:
                    continue
                samples.append([content, label])
            result[data_type] = samples
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_polarities


class ASOTEDataRest14(ASOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'rest14'


class ASOTEDataLapt14(ASOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'lapt14'


class ASOTEDataRest15(ASOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'rest15'


class ASOTEDataRest16(ASOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'rest16'


class ASMOTEData(BaseDataset):
    """
    """

    def __init__(self, configuration: dict={}):
        super().__init__(configuration)

    def to_aspect_tags(self, aspect_term_dict: dict, words: List[str]):
        """
        aspect_term_dict only contains one aspect term
        :param aspect_term_dict:
        :param words:
        :return:
        """
        tags = ['O' for _ in words]
        start = aspect_term_dict['start']
        end = aspect_term_dict['end']
        tags[start] = 'B'
        for i in range(start + 1, end):
            tags[i] = 'I'
        return tags

    def to_opinion_tags(self, opinion_term_dicts: List[Dict], words: List[str]):
        """
        opinion_terms_dict may be contain zero, one, or multiple opinion terms
        :param opinion_term_dicts:
        :param words:
        :return:
        """
        tags = ['O' for _ in words]
        for opinion_term_dict in opinion_term_dicts:
            start = opinion_term_dict['start']
            end = opinion_term_dict['end']
            tags[start] = 'B'
            for i in range(start + 1, end):
                tags[i] = 'I'
        return tags

    def is_include_this_sample(self, polarity):
        if polarity != 'conflict':
            return True
        else:
            try:
                if self.configuration['include_conflict']:
                    return True
            except:
                return False

    def _load_data_by_filepath(self, filepath):
        lines = file_utils.read_all_lines(filepath)
        result = []

        keys = set()
        for line in lines:
            original_line_data = json.loads(line)
            text = original_line_data['sentence']
            words = original_line_data['words']
            polarity = original_line_data['polarity']
            if not self.is_include_this_sample(polarity):
                continue

            aspect_term_str = aspect_term_dict_to_str(original_line_data['aspect_term'])
            key = '%s_%s' % (text, aspect_term_str)
            keys.add(key)

            target_tags = self.to_aspect_tags(original_line_data['aspect_term'], words)

            opinion_words_tags = self.to_opinion_tags(original_line_data['opinions'], words)

            metadata = {'original_line_data': original_line_data}
            sentence = Sentence(text, words, target_tags, opinion_words_tags, polarity=polarity,
                                metadata=metadata)
            result.append(sentence)

        if self.configuration is not None and 'add_predicted_aspect_term' in self.configuration and self.configuration['add_predicted_aspect_term']:
            ate_result_lines = file_utils.read_all_lines(self.configuration['ate_result_filepath'])
            for line in ate_result_lines:
                original_line_data = json.loads(line)
                text: str = original_line_data['text']
                words = text.split(' ')
                polarity = 'positive'
                if not self.is_include_this_sample(polarity):
                    continue

                for aspect_term_str in original_line_data['pred']:
                    key = '%s_%s' % (text, aspect_term_str)
                    if key in keys:
                        continue

                    aspect_term = aspect_term_str_to_dict(aspect_term_str)

                    target_tags = self.to_aspect_tags(aspect_term, words)

                    opinion_words_tags = self.to_opinion_tags([], words)

                    original_line_data_copy = copy.deepcopy(original_line_data)
                    original_line_data_copy['sentence'] = text
                    original_line_data_copy['words'] = copy.deepcopy(words)
                    original_line_data_copy['polarity'] = polarity
                    original_line_data_copy['aspect_term'] = aspect_term
                    original_line_data_copy['opinions'] = []

                    metadata = {'original_line_data': original_line_data_copy}
                    sentence = Sentence(text, original_line_data_copy['words'], target_tags, opinion_words_tags, polarity=polarity,
                                        metadata=metadata)
                    result.append(sentence)
        return result

    def _get_sub_dir(self):
        raise NotImplementedError()

    def _load_train_dev_test_data(self):
        base_dir = os.path.join(base_data_dir,
                                'ASMOTE',
                                 self._get_sub_dir(),
                                 'asmote_gold_standard'
                                )
        train_filepath = os.path.join(base_dir, 'train.txt')
        dev_filepath = os.path.join(base_dir, 'dev.txt')
        test_filepath = os.path.join(base_dir, 'test.txt')

        print('train data:')
        train_data = self._load_data_by_filepath(train_filepath)
        print('dev data:')
        dev_data = self._load_data_by_filepath(dev_filepath)
        print('test data:')
        test_data = self._load_data_by_filepath(test_filepath)

        return train_data, dev_data, test_data

    def generate_tosc_data(self, test_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for sentence in data:
                sentence: Sentence = sentence
                content = sentence.text
                label = []
                words = sentence.words
                for opinion in sentence.metadata['original_line_data']['opinions']:
                    if 'polarity' not in opinion:
                        continue
                    polarity = opinion['polarity']
                    distinct_polarities.add(polarity)

                    term = opinion['opinion_term']['term']
                    word_start = opinion['opinion_term']['start']
                    word_end = opinion['opinion_term']['end']

                    start = 0 if word_start == 0 else len(' '.join(words[: word_start])) + 1
                    end = len(' '.join(words[: word_end]))

                    term_temp = content[start: end]
                    if term_temp != term:
                        print('char index error, term: %s term_temp: %s' % (term, term_temp))

                    opinion_term = AspectTerm(term, polarity, start, end, metadata={'opinion': opinion})
                    label.append(opinion_term)
                if len(label) == 0:
                    continue
                samples.append([content, label])
            result[data_type] = samples
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_polarities


class ASMOTEDataRest14(ASMOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'SemEval-2014-Task-4-REST'


class ASMOTEDataLapt14(ASMOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'SemEval-2014-Task-4-LAPT'


class ASMOTEDataRest15(ASMOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'SemEval-2015-Task-12-REST'


class ASMOTEDataRest16(ASMOTEData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return 'SemEval-2016-Task-5-REST-SB1'


class SemEvalTripletData(BaseDataset):
    """
    Knowing What, How and Why: A Near Complete Solution for Aspect-based Sentiment Analysis
    """

    def __init__(self, configuration: dict={}):
        super().__init__(configuration)

    def _load_data_by_filepath(self, filepath):
        lines = file_utils.read_all_lines(filepath)
        result = []
        target_without_opinions_counter = 0
        for line in lines:
            parts = line.split('####')
            text = parts[0]
            words = text.split(' ')
            target_tags = [tag.split('=')[1] for tag in parts[1].split(' ')]
            opinion_words_tags = [tag.split('=')[1] for tag in parts[2].split(' ')]
            for i in range(1, len(words) + 1):
                polarity = ''
                target_tags_copy = copy.deepcopy(target_tags)
                target_prefix = 'T' * i + '-'
                target_counter = 0
                for j, tag in enumerate(target_tags_copy):
                    if not tag.startswith(target_prefix):
                        target_tags_copy[j] = 'O'
                    else:
                        tag_parts = tag.split('-')
                        polarity = tag_parts[1]
                        if j == 0 or target_tags_copy[j - 1] == 'O':
                            target_tags_copy[j] = 'B'
                            target_counter += 1
                        else:
                            target_tags_copy[j] = 'I'
                if target_counter > 1:
                    print('')
                if target_counter == 0:
                    break
                opinion_words_tags_copy = copy.deepcopy(opinion_words_tags)
                opinion_tag = 'S' * i
                sentiment_counter = 0
                for j, tag in enumerate(opinion_words_tags_copy):
                    if tag != opinion_tag:
                        opinion_words_tags_copy[j] = 'O'
                    else:
                        if j == 0 or opinion_words_tags_copy[j - 1] == 'O':
                            opinion_words_tags_copy[j] = 'B'
                            sentiment_counter += 1
                        else:
                            opinion_words_tags_copy[j] = 'I'
                if sentiment_counter == 0:
                    target_without_opinions_counter += 1
                metadata = {'original_line': line}
                sentence = Sentence(text, copy.deepcopy(words), target_tags_copy, opinion_words_tags_copy, polarity=polarity,
                                    metadata=metadata)
                result.append(sentence)
        print('target_without_opinions_counter: %d' % target_without_opinions_counter)
        return result

    def _get_sub_dir(self):
        raise NotImplementedError()

    def _load_train_dev_test_data(self):
        base_dir = os.path.join(base_data_dir,
                                'SemEval-Triplet-data',
                                'triplet_data_only',
                                self._get_sub_dir())
        train_filepath = os.path.join(base_dir, 'train.txt')
        dev_filepath = os.path.join(base_dir, 'dev.txt')
        test_filepath = os.path.join(base_dir, 'test.txt')

        print('train data:')
        train_data = self._load_data_by_filepath(train_filepath)
        print('dev data:')
        dev_data = self._load_data_by_filepath(dev_filepath)
        print('test data:')
        test_data = self._load_data_by_filepath(test_filepath)

        return train_data, dev_data, test_data


class TripletRest14(SemEvalTripletData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14res'


class TripletLapt14(SemEvalTripletData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14lap'


class TripletRest15(SemEvalTripletData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '15res'


class TripletRest16(SemEvalTripletData):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '16res'


class SemEvalTripletDataV2(BaseDataset):
    """
    2020-emnlp-Position-Aware Tagging for Aspect Sentiment Triplet Extraction
    """

    def __init__(self, configuration: dict={}):
        super().__init__(configuration)

    def _generate_tags(self, tags: List[str], indices: List[int]):
        """

        :param tags:
        :param indices:
        :return:
        """
        tags[indices[0]] = 'B'
        if len(indices) > 1:
            for index in indices[1:]:
                tags[index] = 'I'

    def _load_data_by_filepath(self, filepath):
        lines = file_utils.read_all_lines(filepath)
        sentences = set()
        polarity_counter = defaultdict(int)
        result = []
        for line in lines:
            parts = line.split('####')
            text = parts[0]

            if text not in sentences:
                sentences.add(text)
            else:
                print(line)

            words = text.split(' ')

            labels = eval(parts[1])
            # 针对15和16的数据集，当一个句子对一个aspect term同时表达不同情感的观点时，emnlp2020用标注里最后那个
            # 情感作为所有三元组的观点
            # 示例(16res)
            # Spreads and toppings are great - though a bit pricey.
            # Food was just average...if they lowered the prices just a bit, it would be a bigger draw.
            aspect_sentiment_and_opinions = defaultdict(list)
            for label in labels:
                aspect = tuple(label[0])
                sentiment = label[2]
                opinion = label[1]
                aspect_sentiment_and_opinions[(aspect, sentiment)].append(opinion)
                polarity_counter[sentiment] += 1

            for aspect_sentiment, opinions in aspect_sentiment_and_opinions.items():
                unique_opinion_num = len(set([str(e) for e in opinions]))
                if unique_opinion_num != len(opinions):
                    print('sentence with duplicate opionion: %s' % line)

                aspect = aspect_sentiment[0]
                sentiment = aspect_sentiment[1]
                target_tags = ['O'] * len(words)
                self._generate_tags(target_tags, aspect)
                opinion_words_tags = ['O'] * len(words)
                for opinion in opinions:
                    self._generate_tags(opinion_words_tags, opinion)
                metadata = {'original_line': line}
                sentence = Sentence(text, copy.deepcopy(words), target_tags, opinion_words_tags,
                                    polarity=sentiment,
                                    metadata=metadata)
                result.append(sentence)
        return result

    def _get_sub_dir(self):
        raise NotImplementedError()

    def _load_train_dev_test_data(self):
        base_dir = os.path.join(base_data_dir,
                                'SemEval-Triplet-data',
                                'ASTE-Data-V2-EMNLP2020',
                                self._get_sub_dir())
        train_filepath = os.path.join(base_dir, 'train_triplets.txt')
        dev_filepath = os.path.join(base_dir, 'dev_triplets.txt')
        test_filepath = os.path.join(base_dir, 'test_triplets.txt')

        print('train data:')
        train_data = self._load_data_by_filepath(train_filepath)
        print('dev data:')
        dev_data = self._load_data_by_filepath(dev_filepath)
        print('test data:')
        test_data = self._load_data_by_filepath(test_filepath)

        return train_data, dev_data, test_data


class TripletRest14V2(SemEvalTripletDataV2):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14res'


class TripletLapt14V2(SemEvalTripletDataV2):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14lap'


class TripletRest15V2(SemEvalTripletDataV2):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '15res'


class TripletRest16V2(SemEvalTripletDataV2):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '16res'


class SemEvalTripletDataSupportingSharedOpinion(BaseDataset):
    """
    Knowing What, How and Why: A Near Complete Solution for Aspect-based Sentiment Analysis
    """

    def __init__(self, configuration: dict={}):
        super().__init__(configuration)

    def _load_data_by_filepath(self, filepath, auxiliary_word_target_and_opinions: dict):
        lines = file_utils.read_all_lines(filepath)
        result = []
        opinions_from_naacl_counter = 0
        for line in lines:
            parts = line.split('####')
            text = parts[0]
            words = text.split(' ')
            target_tags = [tag.split('=')[1] for tag in parts[1].split(' ')]
            opinion_words_tags = [tag.split('=')[1] for tag in parts[2].split(' ')]
            for i in range(1, len(words) + 1):
                polarity = ''
                target_tags_copy = copy.deepcopy(target_tags)
                target_prefix = 'T' * i + '-'
                b_counter = 0
                for j, tag in enumerate(target_tags_copy):
                    if not tag.startswith(target_prefix):
                        target_tags_copy[j] = 'O'
                    else:
                        tag_parts = tag.split('-')
                        polarity = tag_parts[1]
                        if j == 0 or target_tags_copy[j - 1] == 'O':
                            target_tags_copy[j] = 'B'
                            b_counter += 1
                        else:
                            target_tags_copy[j] = 'I'
                if b_counter > 1:
                    print('')
                if b_counter == 0:
                    break
                opinion_words_tags_copy = copy.deepcopy(opinion_words_tags)
                opinion_tag = 'S' * i
                for j, tag in enumerate(opinion_words_tags_copy):
                    if tag != opinion_tag:
                        opinion_words_tags_copy[j] = 'O'
                    else:
                        if j == 0 or opinion_words_tags_copy[j - 1] == 'O':
                            opinion_words_tags_copy[j] = 'B'
                        else:
                            opinion_words_tags_copy[j] = 'I'

                if 'B' not in opinion_words_tags_copy:
                    opinions_from_naacl_counter += 1
                    key = ' '.join(['%s\\%s' % (words[i], target_tags_copy[i]) for i in range(len(words))])
                    try:
                        naacl_opinion_words_tags = auxiliary_word_target_and_opinions[key]
                        if opinion_words_tags_copy != naacl_opinion_words_tags:
                            opinion_words_tags_copy = naacl_opinion_words_tags
                    except:
                        print(' '.join(words))
                        print(key)
                        continue

                metadata = {'original_line': line}
                sentence = Sentence(text, copy.deepcopy(words), target_tags_copy, opinion_words_tags_copy, polarity=polarity,
                                    metadata=metadata)
                result.append(sentence)
        print('opinions_from_naacl_counter: %d' % opinions_from_naacl_counter)
        return result

    def _get_sub_dir(self):
        raise NotImplementedError()

    def _get_auxiliary_data(self):
        raise NotImplementedError()

    def _load_train_dev_test_data(self):
        base_dir = os.path.join(base_data_dir,
                                'SemEval-Triplet-data',
                                'triplet_data_only',
                                self._get_sub_dir())
        train_filepath = os.path.join(base_dir, 'train.txt')
        dev_filepath = os.path.join(base_dir, 'dev.txt')
        test_filepath = os.path.join(base_dir, 'test.txt')

        auxiliary_data: BaseDataset = self._get_auxiliary_data()
        data_type_and_data = auxiliary_data.get_data_type_and_data_dict()
        auxiliary_word_target_and_opinions = {}
        for data in data_type_and_data.values():
            for sample in data:
                sample: Sentence = sample
                words = sample.words
                target_tags = sample.target_tags
                opinion_tags = sample.opinion_words_tags
                key = ' '.join(['%s\\%s' % (words[i], target_tags[i]) for i in range(len(words))])
                auxiliary_word_target_and_opinions[key] = opinion_tags

        print('train_data:')
        train_data = self._load_data_by_filepath(train_filepath, auxiliary_word_target_and_opinions)
        print('dev_data:')
        dev_data = self._load_data_by_filepath(dev_filepath, auxiliary_word_target_and_opinions)
        print('test_data:')
        test_data = self._load_data_by_filepath(test_filepath, auxiliary_word_target_and_opinions)

        return train_data, dev_data, test_data


class TripletRest14SupportingSharedOpinion(SemEvalTripletDataSupportingSharedOpinion):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14res'

    def _get_auxiliary_data(self):
        return Rest14()


class TripletLapt14SupportingSharedOpinion(SemEvalTripletDataSupportingSharedOpinion):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '14lap'

    def _get_auxiliary_data(self):
        return Lapt14()


class TripletRest15SupportingSharedOpinion(SemEvalTripletDataSupportingSharedOpinion):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '15res'

    def _get_auxiliary_data(self):
        return Rest15()


class TripletRest16SupportingSharedOpinion(SemEvalTripletDataSupportingSharedOpinion):
    """

    """

    def __init__(self, configuration: Dict=None):
        super().__init__(configuration)

    def _get_sub_dir(self):
        return '16res'

    def _get_auxiliary_data(self):
        return Rest16()


suported_dataset_names_and_data_loader = {
    'rest14': Rest14,  # 2019-naacl-Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling
    'lapt14': Lapt14,
    'rest15': Rest15,
    'rest16': Rest16,
    'triplet_rest14': TripletRest14,
    'triplet_lapt14': TripletLapt14,
    'triplet_rest15': TripletRest15,
    'triplet_rest16': TripletRest16,
    'triplet_rest14_v2': TripletRest14V2,  # 2020-emnlp-Position-Aware Tagging for Aspect Sentiment Triplet Extraction
    'triplet_lapt14_v2': TripletLapt14V2,
    'triplet_rest15_v2': TripletRest15V2,
    'triplet_rest16_v2': TripletRest16V2,
    'triplet_rest14_supporting_shared_opinion': TripletRest14SupportingSharedOpinion,
    'triplet_lapt14_supporting_shared_opinion': TripletLapt14SupportingSharedOpinion,
    'triplet_rest15_supporting_shared_opinion': TripletRest15SupportingSharedOpinion,
    'triplet_rest16_supporting_shared_opinion': TripletRest16SupportingSharedOpinion,
    'ASOTEDataRest14': ASOTEDataRest14,
    'ASOTEDataLapt14': ASOTEDataLapt14,
    'ASOTEDataRest15': ASOTEDataRest15,
    'ASOTEDataRest16': ASOTEDataRest16,
    'ASMOTEDataRest14': ASMOTEDataRest14,
    'ASMOTEDataLapt14': ASMOTEDataLapt14,
    'ASMOTEDataRest15': ASMOTEDataRest15,
    'ASMOTEDataRest16': ASMOTEDataRest16
}


def get_dataset_class_by_name(dataset_name):
    """

    :param dataset_name:
    :return:
    """
    return suported_dataset_names_and_data_loader[dataset_name]


def explore_towe_datasets(merge_train_dev=True):
    """
    2019-naacl-Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling
    :return:
    """
    print('data from 2019-naacl-Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling:')
    dataset_names = ['rest14', 'lapt14', 'rest15', 'rest16']
    for dataset_name in dataset_names:
        print('dataset_name: %s' % dataset_name)
        dataset = get_dataset_class_by_name(dataset_name)()
        data_type_and_data: dict = dataset.get_data_type_and_data_dict()
        if merge_train_dev:
            data_type_and_data['train'].extend(data_type_and_data['dev'])
            data_type_and_data.pop('dev')
        data_type_and_statistics = {}
        for data_type, data in data_type_and_data.items():
            statistics = defaultdict(int)
            sentences = set()
            for sentence in data:
                sentence: Sentence = sentence
                sentences.add(sentence.text)
                aspect_terms = sequence_labeling_utils.terms_from_tags(sentence.target_tags, sentence.words)
                statistics['aspect_term_num'] += len(aspect_terms)
                opinion_terms = sequence_labeling_utils.terms_from_tags(sentence.opinion_words_tags, sentence.words)
                statistics['opinion_term_num'] += len(opinion_terms)
            statistics['sentence_num'] = len(sentences)
            print(data_type)
            print('%d\t%d\t%d' % (statistics['sentence_num'], statistics['aspect_term_num'],
            statistics['opinion_term_num']))
            data_type_and_statistics[data_type] = statistics
        print(dict(data_type_and_statistics))


def explore_aste_data_v2_datasets(merge_train_dev=False):
    """
    2020-emnlp-Position-Aware Tagging for Aspect Sentiment Triplet Extraction
    :return:
    """
    print('data from 2020-emnlp-Position-Aware Tagging for Aspect Sentiment Triplet Extraction:')
    dataset_names = ['triplet_rest14_v2', 'triplet_lapt14_v2', 'triplet_rest15_v2', 'triplet_rest16_v2']
    for dataset_name in dataset_names:
        print('dataset_name: %s' % dataset_name)
        dataset = get_dataset_class_by_name(dataset_name)()
        data_type_and_data: dict = dataset.get_data_type_and_data_dict()
        if merge_train_dev:
            data_type_and_data['train'].extend(data_type_and_data['dev'])
            data_type_and_data.pop('dev')
        data_type_and_statistics = {}
        for data_type, data in data_type_and_data.items():
            statistics = defaultdict(int)
            sentences = set()
            for sentence in data:
                sentence: Sentence = sentence
                sentences.add(sentence.text)
                aspect_terms = sequence_labeling_utils.terms_from_tags(sentence.target_tags, sentence.words)
                statistics['aspect_term_num'] += len(aspect_terms)
                opinion_terms = sequence_labeling_utils.terms_from_tags(sentence.opinion_words_tags, sentence.words)
                statistics['opinion_term_num'] += len(opinion_terms)
                statistics[sentence.polarity] += len(opinion_terms)
            statistics['sentence_num'] = len(sentences)
            print(data_type)
            print('%d\t%d\t%d\t%d\t%d\t%d' % (statistics['sentence_num'], statistics['aspect_term_num'],
            statistics['opinion_term_num'], statistics['POS'], statistics['NEU'], statistics['NEG']))
            data_type_and_statistics[data_type] = statistics
        print(dict(data_type_and_statistics))


def explore_asmote_datasets(merge_train_dev=False):
    """
    ASMOTEDataRest16
    :return:
    """
    print('ASMOTE')
    dataset_names = ['ASMOTEDataRest14', 'ASMOTEDataLapt14', 'ASMOTEDataRest15', 'ASMOTEDataRest16']
    configuration = {'include_conflict': True}
    for dataset_name in dataset_names:
        print('dataset_name: %s' % dataset_name)
        dataset = get_dataset_class_by_name(dataset_name)(configuration)
        data_type_and_data: dict = dataset.get_data_type_and_data_dict()
        if merge_train_dev:
            data_type_and_data['train'].extend(data_type_and_data['dev'])
            data_type_and_data.pop('dev')
        data_type_and_statistics = {}
        for data_type, data in data_type_and_data.items():
            print('data_type: %s' % data_type)
            statistics = defaultdict(int)
            sentences = set()
            no_opinion_target_num = 0
            one_opinion_target_num = 0
            more_than_one_opinion_target_num = 0
            for sentence in data:
                sentence: Sentence = sentence
                sentences.add(sentence.text)
                aspect_terms = sequence_labeling_utils.terms_from_tags(sentence.target_tags, sentence.words)
                if len(aspect_terms) != 1:
                    print('error')
                statistics['aspect_term_num'] += len(aspect_terms)
                opinion_terms = sequence_labeling_utils.terms_from_tags(sentence.opinion_words_tags, sentence.words)
                statistics[sentence.polarity] += 1
                if len(opinion_terms) == 0:
                    no_opinion_target_num += 1
                elif len(opinion_terms) == 1:
                    one_opinion_target_num += 1
                else:
                    more_than_one_opinion_target_num += 1
                if len(opinion_terms) > 0:
                    statistics['triplet_num'] += 1
            statistics['sentence_num'] = len(sentences)
            # print(data_type)
            print('%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d' % (statistics['sentence_num'], statistics['aspect_term_num'],
            statistics['triplet_num'], statistics['positive'], statistics['neutral'], statistics['negative'],
            statistics['conflict'], no_opinion_target_num, one_opinion_target_num, more_than_one_opinion_target_num))
            data_type_and_statistics[data_type] = statistics
        print(dict(data_type_and_statistics))


def explore_asote_datasets(merge_train_dev=False):
    """
    ASMOTEDataRest16
    :return:
    """
    print('ASOTE')
    dataset_names = ['ASOTEDataRest14', 'ASOTEDataLapt14', 'ASOTEDataRest15', 'ASOTEDataRest16']
    configuration = {'include_conflict': False}
    for dataset_name in dataset_names:
        print('dataset_name: %s' % dataset_name)
        dataset = get_dataset_class_by_name(dataset_name)(configuration)
        data_type_and_data: dict = dataset.get_data_type_and_data_dict()
        if merge_train_dev:
            data_type_and_data['train'].extend(data_type_and_data['dev'])
            data_type_and_data.pop('dev')
        data_type_and_statistics = {}
        for data_type, data in data_type_and_data.items():
            print('data_type: %s' % data_type)
            statistics = defaultdict(int)
            sentences = set()
            for sentence in data:
                sentence: Sentence = sentence
                sentences.add(sentence.text)
                statistics['aspect_term_num'] += 1

                original_line_data = sentence.metadata['original_line_data']
                opinions = original_line_data['opinions']
                unique_sentiments = set()
                valid_opinions = []
                for opinion_term in opinions:
                    if 'polarity' in opinion_term:
                        unique_sentiments.add(opinion_term['polarity'])
                        valid_opinions.append(opinion_term)
                if len(unique_sentiments) > 1:
                    statistics['aspect_with_different_sentiments'] += 1
                if len(valid_opinions) == 0:
                    statistics['aspect_with_no_triplets'] += 1
                elif len(valid_opinions) == 1:
                    statistics['aspect_with_one_triplets'] += 1
                    if valid_opinions[0]['polarity'] != sentence.polarity and sentence.polarity != 'conflict':
                        statistics['aspect_sentiment_different_from_opinion'] += 1
                else:
                    statistics['aspect_with_multi_triplets'] += 1

                statistics['triplets_num'] += len(valid_opinions)
            statistics['sentence_num'] = len(sentences)
            # print(data_type)
            print('%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d' % (statistics['sentence_num'],
                                                  statistics['aspect_term_num'],
                                                  statistics['triplets_num'],
                                                  statistics['aspect_with_no_triplets'],
                                                  statistics['aspect_with_one_triplets'],
                                                  statistics['aspect_with_multi_triplets'],
                                                  statistics['aspect_with_different_sentiments'],
                                                      statistics['aspect_sentiment_different_from_opinion']
                                                  ))
            data_type_and_statistics[data_type] = statistics
        print(dict(data_type_and_statistics))


def find_examples_from_asmote_datasets(merge_train_dev=False):
    """
    ASMOTEDataRest16
    :return:
    """
    print('ASMOTE')
    dataset_names = ['ASMOTEDataRest14', 'ASMOTEDataLapt14', 'ASMOTEDataRest15', 'ASMOTEDataRest16']
    configuration = {'include_conflict': False}
    for dataset_name in dataset_names:
        print('dataset_name: %s' % dataset_name)
        dataset = get_dataset_class_by_name(dataset_name)(configuration)
        data_type_and_data: dict = dataset.get_data_type_and_data_dict()
        if merge_train_dev:
            data_type_and_data['train'].extend(data_type_and_data['dev'])
            data_type_and_data.pop('dev')
        for data_type, data in data_type_and_data.items():
            print('data_type: %s' % data_type)
            sentence_and_aspect_terms = defaultdict(list)
            for sentence in data:
                sentence: Sentence = sentence
                sentence_and_aspect_terms[sentence.text].append(sentence)
            for sentence, aspect_terms in sentence_and_aspect_terms.items():
                if len(sentence) > 100:
                    continue
                if len(aspect_terms) != 2:
                    continue
                polarities = set([e.polarity for e in aspect_terms])
                if len(polarities) < 2:
                    continue
                print(sentence)



if __name__ == '__main__':
    # dataset_names = ['RealASOTripletRest16']
    # for dataset_name in dataset_names:
    #     print('dataset_name: %s' % dataset_name)
    #     dataset = get_dataset_class_by_name(dataset_name)()
    #     data_type_and_data = dataset.get_data_type_and_data_dict()
    #     data_type_and_statistics = {}
    #     for data_type, data in data_type_and_data.items():
    #         statistics = defaultdict(int)
    #         for sentence in data:
    #             sentence: Sentence = sentence
    #             opinions = sentence.metadata['original_line_data']['opinions']
    #             for opinion in opinions:
    #                 if 'opinion_term' not in opinion:
    #                     continue
    #                 statistics[opinion['polarity']] += 1
    #         data_type_and_statistics[data_type] = statistics
    #     print(data_type_and_statistics)
    # explore_towe_datasets()
    # explore_aste_data_v2_datasets()
    explore_asmote_datasets()
    # explore_asote_datasets()
    # find_examples_from_asmote_datasets()