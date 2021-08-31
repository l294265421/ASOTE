
import logging
import os
import json
import re
import csv
import pickle
import sys
from collections import defaultdict
from typing import List
import copy
import collections
from typing import Dict

from bs4 import BeautifulSoup
from bs4.element import Tag
from sklearn.model_selection import train_test_split

from nlp_tasks.common import common_path
from nlp_tasks.utils import file_utils


logger = logging.getLogger(__name__)
base_data_dir = common_path.get_task_data_dir('absa', is_original=True)


class AspectTerm:
    """
    aspect term
    """

    def __init__(self, term, polarity, from_index, to_index, category=None):
        self.term = term
        self.polarity = polarity
        # inclusive
        self.from_index = int(from_index)
        # exclusive
        self.to_index = int(to_index)
        self.category = category

    def __str__(self):
        return '%s-%s-%s-%s-%s' % (self.term, str(self.polarity), str(self.from_index), str(self.to_index),
                                   self.category)


class AspectCategory:
    """
    aspect category
    """

    def __init__(self, category, polarity):
        self.category = category
        self.polarity = polarity

    def __str__(self):
        return '%s-%s' % (self.category, str(self.polarity))


class Text:
    """

    """

    def __init__(self, text, polarity, sample_id=''):
        self.text = text
        self.polarity = polarity
        self.sample_id = sample_id

    def __str__(self):
        return '%s-%s' % (self.text, str(self.polarity))


class AbsaText(Text):
    """

    """

    def __init__(self, text, polarity, aspect_categories, aspect_terms, sample_id=''):
        super().__init__(text, polarity, sample_id=sample_id)
        self.aspect_categories = aspect_categories
        self.aspect_terms = aspect_terms


class AbsaSentence(AbsaText):
    """

    """

    def __init__(self, text, polarity, aspect_categories, aspect_terms, sample_id='', start_index_in_doc=-1):
        super().__init__(text, polarity, aspect_categories, aspect_terms, sample_id=sample_id)
        self.start_index_in_doc = start_index_in_doc


class AbsaDocument(AbsaText):
    """

    """

    def __init__(self, text, polarity, aspect_categories, aspect_terms, absa_sentences, sample_id=''):
        super().__init__(text, polarity, aspect_categories, aspect_terms, sample_id=sample_id)
        self.absa_sentences = absa_sentences

    def get_plain_text_of_sentences(self):
        """

        :return:
        """
        result = []
        if self.absa_sentences is None:
            return result
        for sentence in self.absa_sentences:
            result.append(sentence.text)
        return result


class BaseDataset:
    """
    base class
    memory mirror of datasets
    """

    def __init__(self, configuration: dict=None):
        self.configuration = configuration
        self.train_data, self.dev_data, self.test_data = self._load_train_dev_test_data()

    def _load_train_dev_test_data(self):
        """

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

    def get_sentences(self, data_type: str):
        """

        :param data_type: train, dev, or test
        :return: all sentences in the specified dataset
        """

        data_type_and_data = self.get_data_type_and_data_dict()
        if data_type is None or data_type not in data_type_and_data:
            logger.info('unknown data type: %s' % str(data_type))
            return []
        data = data_type_and_data[data_type]
        sentences = []
        for document in data:
            for sentence in document.get_plain_text_of_sentences():
                sentences.append(sentence)
        return sentences

    def get_documents(self, data_type: str):
        """

        :param data_type: train, dev, or test
        :return: all sentences in the specified dataset
        """

        data_type_and_data = self.get_data_type_and_data_dict()
        if data_type is None or data_type not in data_type_and_data:
            logger.info('unknown data type: %s' % str(data_type))
            return []
        data = data_type_and_data[data_type]
        documents = []
        for document in data:
            documents.append(document.text)
        return documents

    def generate_atsa_data(self, test_size=0.2):
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
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', '', sentence.text)
                    # content = re.sub('[\-/]', ' ', content)
                    label = []
                    for aspect_term in sentence.aspect_terms:
                        label.append(aspect_term)
                        polarity = aspect_term.polarity
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        if result['dev'] is None and test_size is not None:
            original_train_samples = result['train']
            train_samples, dev_samples = train_test_split(original_train_samples, test_size=test_size)
            result['train'] = train_samples
            result['dev'] = dev_samples
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_polarities

    def generate_dev_data(self, result, dev_size, random_state=1234):
        """
        ，
        :param result: data_type_and_data
        :param dev_size:
        :param random_state:
        :return:
        """

        if result['dev'] is None:
            if dev_size != 0.0:
                original_train_samples = result['train']
                train_samples, dev_samples = train_test_split(original_train_samples, test_size=dev_size,
                                                              random_state=random_state)
                result['train'] = train_samples
                result['dev'] = dev_samples
            else:
                result['dev'] = result['test']


class AsgcnData(BaseDataset):
    """
    Aspect-basedSentimentClassiﬁcationwithAspect-speciﬁcGraph ConvolutionalNetworks

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data_by_filepath(self, train_filepath, test_filepath):
        data_type_and_filepath = {'train': train_filepath,
                                  'test': test_filepath}
        data_type_and_data = {}
        for data_type, filepath in data_type_and_filepath.items():
            lines = file_utils.read_all_lines(filepath)
            sentences = []
            polarity_mapping = {'-1': 'negative',
                                '0': 'neutral',
                                '1': 'positive'}
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                polarity = lines[i + 2].strip()
                if text_left != '':
                    text = text_left + " " + aspect
                    from_index = len(text_left) + 1
                else:
                    text = aspect
                    from_index = 0
                if text_right != '':
                    text = text + ' ' + text_right
                to_index = from_index + len(aspect)
                if text[from_index: to_index] != aspect:
                    logger.error('error aspect index: %s != %s' (text[from_index: to_index], aspect))
                aspect_term = AspectTerm(aspect, polarity_mapping[polarity], from_index, to_index)
                sentence = AbsaSentence(text, None, None, [aspect_term])
                sentences.append(sentence)
            documents = [AbsaDocument(sentence.text, None, None, None, [sentence]) for sentence in sentences]
            data_type_and_data[data_type] = documents
        return data_type_and_data['train'], None, data_type_and_data['test']


class AsgcnData2014Rest(AsgcnData):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval14', 'restaurant_train.raw')
        test_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval14', 'restaurant_test.raw')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class AsgcnData2014Lapt(AsgcnData):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval14', 'laptop_train.raw')
        test_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval14', 'laptop_test.raw')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class AsgcnData2015Rest(AsgcnData):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval15', 'restaurant_train.raw')
        test_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval15', 'restaurant_test.raw')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class AsgcnData2016Rest(AsgcnData):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval16', 'restaurant_train.raw')
        test_filepath = os.path.join(base_data_dir, 'ASGCN', 'semeval16', 'restaurant_test.raw')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class YelpDataset(BaseDataset):
    """

    """

    def __init__(self, configuration: dict=None):
        if 'max_word_num' not in configuration:
            configuration['max_word_num'] = sys.maxsize
        if 'max_sample_num_per_class' not in configuration:
            configuration['max_sample_num_per_class'] = sys.maxsize
        super().__init__(configuration)

    def _load_train_dev_test_data_by_filepath(self, train_filepath, test_filepath=None, val_filepath=None):
        """

        :return:
        """
        data_type_and_datas = {}
        data_type_and_filepath = {
            'train': train_filepath,
            'test': test_filepath,
            'dev': val_filepath
        }
        for data_type, filepath in data_type_and_filepath.items():
            if filepath is None:
                data_type_and_datas[data_type] = None
                continue
            lines = file_utils.read_all_lines_generator(filepath)
            documents = []
            polarity_count = defaultdict(int)
            for i, line in enumerate(lines):
                # if i < 100000:
                #     continue
                # ，
                line_dict = json.loads(line)
                text: str = line_dict['text']
                if len(text.split()) > self.configuration['max_word_num']:
                    continue
                stars = line_dict['stars']
                if stars > 3:
                    label = 'positive'
                elif stars == 3:
                    label = 'neutral'
                else:
                    label = 'negative'
                # 2018-Exploiting Document Knowledge for Aspect-level Sentiment Classification 30k
                if polarity_count[label] > self.configuration['max_sample_num_per_class']:
                    continue
                else:
                    polarity_count[label] += 1
                document = AbsaDocument(text, label, None, None, None)
                documents.append(document)
            data_type_and_datas[data_type] = documents
        train_data = data_type_and_datas['train']
        train_data, dev_data = train_test_split(train_data, test_size=0.2)
        test_data = dev_data
        return train_data, dev_data, test_data

    def _load_train_dev_test_data(self):
        train_filepath = self.configuration['train_filepath']
        return self._load_train_dev_test_data_by_filepath(train_filepath)


class Nlpcc2012WeiboSa(BaseDataset):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        data_dir = os.path.join(base_data_dir, 'NLP-CC2012-', '', '')
        filenames = os.listdir(data_dir)
        samples = []
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            content = file_utils.read_all_content(filepath, encoding='utf-16-le')
            soup = BeautifulSoup(content, "lxml")
            weibo_tags: List[Tag] = soup.find_all('weibo')
            for weibo_tag in weibo_tags:
                weibo_id = weibo_tag.get('id', '')
                sentences: List[AbsaSentence] = []
                sentence_tags: List[Tag] = weibo_tag.find_all('sentence')
                sentence_start_index = 0
                for sentence_tag in sentence_tags:
                    sentence_id = sentence_tag.get('id', '')
                    opinionated = sentence_tag.get('opinionated', '')
                    sentence_polarity = None
                    sentence_text = sentence_tag.get_text()
                    aspect_terms = []
                    if opinionated == 'Y':
                        sentence_polarity = sentence_tag.get('polarity')
                        aspect_term_cursor = 1
                        while sentence_tag.get('target_word_%d' % aspect_term_cursor, ''):
                            target_word = sentence_tag.get('target_word_%d' % aspect_term_cursor)
                            target_begin = int(sentence_tag.get('target_begin_%d' % aspect_term_cursor))
                            target_end = int(sentence_tag.get('target_end_%d' % aspect_term_cursor)) + 1
                            target_polarity = sentence_tag.get('target_polarity_%d' % aspect_term_cursor)
                            aspect_term = AspectTerm(target_word, target_polarity, target_begin, target_end)
                            aspect_terms.append(aspect_term)
                            aspect_term_cursor += 1
                    sentence = AbsaSentence(sentence_text, sentence_polarity, None, aspect_terms, sentence_id,
                                            start_index_in_doc=sentence_start_index)
                    sentence_start_index += len(sentence_text)
                    sentences.append(sentence)
                weibo_text = ''.join([sentence.text for sentence in sentences])
                # for aspect_term in aspect_terms:
                #     temp = weibo_text[aspect_term.from_index: aspect_term.to_index]
                weibo = AbsaDocument(weibo_text, None, None, None, sentences, weibo_id)
                samples.append(weibo)
        samples_train, samples_dev = train_test_split(samples, test_size=0.2, random_state=self.configuration['seed'])

        samples_test = []
        test_data_dir = os.path.join(base_data_dir, 'NLP-CC2012-', '',
                                     'sonar-weibo-processed')
        test_filenames = os.listdir(test_data_dir)
        for test_filename in test_filenames:
            if test_filename.endswith('ann'):
                continue
            test_filepath = os.path.join(test_data_dir, test_filename)
            test_filepath_ann = re.sub('txt', 'ann', test_filepath)
            test_lines_temp = file_utils.read_all_lines(test_filepath)[1:]
            test_line_end_index = len(test_lines_temp)
            for i, line in enumerate(test_lines_temp):
                if 'root_text_begin' in line:
                    test_line_end_index = i
                    break
            test_lines = test_lines_temp[: test_line_end_index]
            test_ann_lines = file_utils.read_all_lines(test_filepath_ann)
            opinion_snippets = [line.split('\t')[2] for line in test_ann_lines if len(line) != 0]
            sentences = []
            for test_line in test_lines:
                polarity = None
                for opinion_snippet in opinion_snippets:
                    if opinion_snippet in test_line:
                        polarity = 'other'
                        break
                sentence = AbsaSentence(test_line, polarity, None, None)
                sentences.append(sentence)
            weibo_text = ''.join([sentence.text for sentence in sentences])
            # for aspect_term in aspect_terms:
            #     temp = weibo_text[aspect_term.from_index: aspect_term.to_index]
            weibo = AbsaDocument(weibo_text, None, None, None, sentences)
            samples_test.append(weibo)

        return samples_train, samples_dev, samples_test


class FeedComment(BaseDataset):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        data_dir = os.path.join(base_data_dir, 'NLP-CC2012-', '', '')
        filenames = os.listdir(data_dir)
        samples = []
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            content = file_utils.read_all_content(filepath, encoding='utf-16-le')
            soup = BeautifulSoup(content, "lxml")
            weibo_tags: List[Tag] = soup.find_all('weibo')
            for weibo_tag in weibo_tags:
                weibo_id = weibo_tag.get('id', '')
                sentences: List[AbsaSentence] = []
                sentence_tags: List[Tag] = weibo_tag.find_all('sentence')
                sentence_start_index = 0
                for sentence_tag in sentence_tags:
                    sentence_id = sentence_tag.get('id', '')
                    opinionated = sentence_tag.get('opinionated', '')
                    sentence_polarity = None
                    sentence_text = sentence_tag.get_text()
                    aspect_terms = []
                    if opinionated == 'Y':
                        sentence_polarity = sentence_tag.get('polarity')
                        aspect_term_cursor = 1
                        while sentence_tag.get('target_word_%d' % aspect_term_cursor, ''):
                            target_word = sentence_tag.get('target_word_%d' % aspect_term_cursor)
                            target_begin = int(sentence_tag.get('target_begin_%d' % aspect_term_cursor))
                            target_end = int(sentence_tag.get('target_end_%d' % aspect_term_cursor)) + 1
                            target_polarity = sentence_tag.get('target_polarity_%d' % aspect_term_cursor)
                            aspect_term = AspectTerm(target_word, target_polarity, target_begin, target_end)
                            aspect_terms.append(aspect_term)
                            aspect_term_cursor += 1
                    sentence = AbsaSentence(sentence_text, sentence_polarity, None, aspect_terms, sentence_id,
                                            start_index_in_doc=sentence_start_index)
                    sentence_start_index += len(sentence_text)
                    sentences.append(sentence)
                weibo_text = ''.join([sentence.text for sentence in sentences])
                # for aspect_term in aspect_terms:
                #     temp = weibo_text[aspect_term.from_index: aspect_term.to_index]
                weibo = AbsaDocument(weibo_text, None, None, None, sentences, weibo_id)
                samples.append(weibo)
        # samples_train, samples_dev = train_test_split(samples, test_size=0.2, random_state=self.configuration['seed'])

        samples_test = []
        test_data_dir = os.path.join(base_data_dir, 'NLP-CC2012-', '',
                                     'feed_comment-processed')
        test_filenames = os.listdir(test_data_dir)
        for test_filename in test_filenames:
            if test_filename.endswith('ann'):
                continue
            test_filepath = os.path.join(test_data_dir, test_filename)
            test_filepath_ann = re.sub('txt', 'ann', test_filepath)
            test_lines_temp = file_utils.read_all_lines(test_filepath)
            test_line_end_index = len(test_lines_temp)
            for i, line in enumerate(test_lines_temp):
                if 'root_text_begin' in line:
                    test_line_end_index = i
                    break
            test_lines = test_lines_temp[: test_line_end_index]
            test_ann_lines = file_utils.read_all_lines(test_filepath_ann)
            opinion_snippets = [line.split('\t')[2] for line in test_ann_lines if len(line) != 0]
            sentences = []
            for test_line in test_lines:
                polarity = None
                for opinion_snippet in opinion_snippets:
                    if opinion_snippet in test_line:
                        polarity = 'other'
                        break
                sentence = AbsaSentence(test_line, polarity, None, None)
                sentences.append(sentence)
            weibo_text = ''.join([sentence.text for sentence in sentences])
            # for aspect_term in aspect_terms:
            #     temp = weibo_text[aspect_term.from_index: aspect_term.to_index]
            weibo = AbsaDocument(weibo_text, None, None, None, sentences)
            samples_test.append(weibo)

        samples_train, samples_dev = train_test_split(samples_test, test_size=0.2, random_state=self.configuration['seed'])
        return samples_train, samples_dev, samples_dev


class Semeval2014Task4(BaseDataset):
    """

    """

    def _load_semeval_by_filepath(self, train_filepath, test_filepath, val_filepath=None):
        """

        :return:
        """
        data_type_and_datas = {}
        data_type_and_filepath = {
            'train': train_filepath,
            'test': test_filepath,
            'dev': val_filepath
        }
        for data_type, filepath in data_type_and_filepath.items():
            if filepath is None:
                data_type_and_datas[data_type] = None
                continue
            content = file_utils.read_all_content(filepath)
            soup = BeautifulSoup(content, "lxml")
            sentence_tags = soup.find_all('sentence')
            sentences = []
            for sentence_tag in sentence_tags:
                text = sentence_tag.text
                aspect_term_tags = sentence_tag.find_all('aspectterm')
                aspect_terms = []
                for aspect_term_tag in aspect_term_tags:
                    term = aspect_term_tag['term']
                    try:
                        polarity = aspect_term_tag['polarity']
                    except:
                        polarity = 'positive'
                    from_index = aspect_term_tag['from']
                    to_index = aspect_term_tag['to']
                    aspect_term = AspectTerm(term, polarity, from_index, to_index)
                    aspect_terms.append(aspect_term)
                aspect_categories = []
                aspect_category_tags = sentence_tag.find_all('aspectcategory')
                for aspect_category_tag in aspect_category_tags:
                    category = aspect_category_tag['category']
                    try:
                        polarity = aspect_category_tag['polarity']
                    except:
                        polarity = 'positive'
                    aspect_category = AspectCategory(category, polarity)
                    aspect_categories.append(aspect_category)
                sentence = AbsaSentence(text, None, aspect_categories, aspect_terms)
                sentences.append(sentence)
            documents = [AbsaDocument(sentence.text, None, None, None, [sentence]) for sentence in sentences]
            data_type_and_datas[data_type] = documents
        train_data = data_type_and_datas['train']
        dev_data = data_type_and_datas['dev']
        test_data = data_type_and_datas['test']
        return train_data, dev_data, test_data


class Semeval2014Task4Lapt(Semeval2014Task4):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-LAPT', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'conceptnet_augment_data.pkl')

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-LAPT', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'Laptop_Train_v2.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-LAPT', 'origin',
                                     "ABSA_Gold_TestData",
                                     'Laptops_Test_Gold.xml')
        return super()._load_semeval_by_filepath(train_filepath, test_filepath)


class Semeval2014Task4RestDevSplits(Semeval2014Task4):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'conceptnet_augment_data.pkl')

    def _load_train_dev_test_data(self):
        sentence_map_filepath = os.path.join(base_data_dir, 'ABSA_DevSplits', 'dataset',
                                             'sentence_map.txt')
        sentence_map = {line.split('\t')[0]: line.split('\t')[1] for line in
                        file_utils.read_all_lines(sentence_map_filepath, strip_type='line_separator')}

        data_filepath = os.path.join(base_data_dir, 'ABSA_DevSplits', 'dataset',
                                      'Restaurants_category.pkl')
        polarity_index_and_text = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }
        datasets = []
        with open(data_filepath, mode='rb') as in_file:
            content = in_file.read()
            content_correct = b''
            for line in content.splitlines():
                content_correct += line + str.encode('\n')
            data = pickle.loads(content_correct, encoding='utf-8')
            # data = pickle.load(in_file, encoding='utf-8')
            datasets_indexed = [data['train'], data['dev'], data['test']]
            index2word = data['index_word']
            for dataset_indexed in datasets_indexed:
                dataset = []
                text_and_categories = {}
                for sample in dataset_indexed:
                    words = [index2word[index] for index in sample[0]]
                    text = ' '.join(words)
                    category = [index2word[index] for index in sample[2]][0]
                    polarity = polarity_index_and_text[sample[4]]
                    aspect_category = AspectCategory(category, polarity)
                    if text not in text_and_categories:
                        text_and_categories[text] = []
                    text_and_categories[text].append(aspect_category)
                for text, categories in text_and_categories.items():
                    text = sentence_map[text]
                    sentence = AbsaSentence(text, None, categories, None)
                    document = AbsaDocument(sentence.text, None, None, None, [sentence])
                    dataset.append(document)
                datasets.append(dataset)

        return datasets

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2014Task4RestDevSplitsAspectTerm(Semeval2014Task4):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def clean_str(self, sentence: str):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """

        sentence_clean = sentence.lower()
        sentence_clean = re.sub('[!"#$%&\'\-()*+,./:;<=>?@[\\]^_`{|}~]', '', sentence_clean)
        sentence_clean = sentence_clean.strip()
        sentence_clean = re.sub('\s', '', sentence_clean)
        return sentence_clean

    def _load_train_dev_test_data(self):
        rest14 = Semeval2014Task4Rest()
        test_with_conflict = rest14.test_data
        test = []
        for sample in test_with_conflict:
            aspect_terms = []
            for aspect_term in sample.absa_sentences[0].aspect_terms:
                if aspect_term.polarity != 'conflict':
                    aspect_terms.append(aspect_term)
            if len(aspect_terms) == 0:
                continue
            sample.absa_sentences[0].aspect_terms = aspect_terms
            test.append(sample)
        all_train_with_conflict = rest14.train_data
        all_train = []
        for sample in all_train_with_conflict:
            aspect_terms = []
            for aspect_term in sample.absa_sentences[0].aspect_terms:
                if aspect_term.polarity != 'conflict':
                    aspect_terms.append(aspect_term)
            if len(aspect_terms) == 0:
                continue
            sample.absa_sentences[0].aspect_terms = aspect_terms
            all_train.append(sample)

        data_filepath = os.path.join(base_data_dir, 'ABSA_DevSplits', 'dataset',
                                      'Restaurants_term.pkl')

        sentence_and_data_type = {}
        with open(data_filepath, mode='rb') as in_file:
            content = in_file.read()
            content_correct = b''
            for line in content.splitlines():
                content_correct += line + str.encode('\n')
            data = pickle.loads(content_correct, encoding='utf-8')
            # data = pickle.load(in_file, encoding='utf-8')
            datasets_indexed = {'train': data['train'], 'dev': data['dev']}
            index2word = data['index_word']
            for data_type, dataset_indexed in datasets_indexed.items():
                for sample in dataset_indexed:
                    words = [index2word[index] for index in sample[0]]
                    text = ' '.join(words)
                    text_clean = re.sub('\s', '', text)
                    sentence_and_data_type[text_clean] = data_type

        train = []
        dev = []
        for sample in all_train:
            sentence = sample.text
            sentence_clean = self.clean_str(sentence)
            if sentence_clean not in sentence_and_data_type:
                if len(sample.absa_sentences[0].aspect_terms) != 0:
                    print('%s is not in sentence_and_data_type' % sentence.strip())
                continue
            data_type = sentence_and_data_type[sentence_clean]
            if data_type == 'train':
                train.append(sample)
            else:
                dev.append(sample)

        return train, dev, test


class Semeval2014Task4LaptDevSplitsAspectTerm(Semeval2014Task4):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def clean_str(self, sentence: str):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """

        sentence_clean = sentence.lower()
        sentence_clean = re.sub('[!"#$%&\'\-()*+,./:;<=>?@[\\]^_`{|}~]', '', sentence_clean)
        sentence_clean = sentence_clean.strip()
        sentence_clean = re.sub('\s', '', sentence_clean)
        return sentence_clean

    def _load_train_dev_test_data(self):
        latp14 = Semeval2014Task4Lapt()
        test_with_conflict = latp14.test_data
        test = []
        for sample in test_with_conflict:
            aspect_terms = []
            for aspect_term in sample.absa_sentences[0].aspect_terms:
                if aspect_term.polarity != 'conflict':
                    aspect_terms.append(aspect_term)
            if len(aspect_terms) == 0:
                continue
            sample.absa_sentences[0].aspect_terms = aspect_terms
            test.append(sample)
        all_train_with_conflict = latp14.train_data
        all_train = []
        for sample in all_train_with_conflict:
            aspect_terms = []
            for aspect_term in sample.absa_sentences[0].aspect_terms:
                if aspect_term.polarity != 'conflict':
                    aspect_terms.append(aspect_term)
            if len(aspect_terms) == 0:
                continue
            sample.absa_sentences[0].aspect_terms = aspect_terms
            all_train.append(sample)

        data_filepath = os.path.join(base_data_dir, 'ABSA_DevSplits', 'dataset',
                                      'Laptop_term.pkl')

        sentence_and_data_type = {}
        with open(data_filepath, mode='rb') as in_file:
            content = in_file.read()
            content_correct = b''
            for line in content.splitlines():
                content_correct += line + str.encode('\n')
            data = pickle.loads(content_correct, encoding='utf-8')
            # data = pickle.load(in_file, encoding='utf-8')
            datasets_indexed = {'train': data['train'], 'dev': data['dev']}
            index2word = data['index_word']
            for data_type, dataset_indexed in datasets_indexed.items():
                for sample in dataset_indexed:
                    words = [index2word[index] for index in sample[0]]
                    text = ' '.join(words)
                    text_clean = re.sub('\s', '', text)
                    sentence_and_data_type[text_clean] = data_type

        train = []
        dev = []
        for sample in all_train:
            sentence = sample.text
            sentence_clean = self.clean_str(sentence)
            if sentence_clean not in sentence_and_data_type:
                if len(sample.absa_sentences[0].aspect_terms) != 0:
                    print('%s is not in sentence_and_data_type' % sentence.strip())
                continue
            data_type = sentence_and_data_type[sentence_clean]
            if data_type == 'train':
                train.append(sample)
            else:
                dev.append(sample)

        return train, dev, test


class Se1415CategoryDevSplits(Semeval2014Task4):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'conceptnet_augment_data.pkl')

    def _load_train_dev_test_data(self):
        data_filepath = os.path.join(base_data_dir, 'ABSA_DevSplits', 'dataset',
                                      'SE1415_category.pkl')
        polarity_index_and_text = {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }
        datasets = []
        with open(data_filepath, mode='rb') as in_file:
            content = in_file.read()
            content_correct = b''
            for line in content.splitlines():
                content_correct += line + str.encode('\n')
            data = pickle.loads(content_correct, encoding='utf-8')
            # data = pickle.load(in_file, encoding='utf-8')
            datasets_indexed = [data['train'], data['dev'], data['test']]
            index2word = data['index_word']
            for dataset_indexed in datasets_indexed:
                dataset = []
                text_and_categories = {}
                for sample in dataset_indexed:
                    words = [index2word[index] for index in sample[0]]
                    text = ' '.join(words)
                    category = [index2word[index] for index in sample[2]][0]
                    polarity = polarity_index_and_text[sample[4]]
                    aspect_category = AspectCategory(category, polarity)
                    if text not in text_and_categories:
                        text_and_categories[text] = []
                    text_and_categories[text].append(aspect_category)
                for text, categories in text_and_categories.items():
                    sentence = AbsaSentence(text, None, categories, None)
                    document = AbsaDocument(sentence.text, None, None, None, [sentence])
                    dataset.append(document)
                datasets.append(dataset)
        return datasets

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2014Task4Rest(Semeval2014Task4):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'conceptnet_augment_data.pkl')

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'Restaurants_Train_v2.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST', 'origin',
                                     "ABSA_Gold_TestData",
                                     'Restaurants_Test_Gold.xml')
        result = super()._load_semeval_by_filepath(train_filepath, test_filepath)
        return result

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class TWITTER(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir,
                                      'Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines',
                                      "data_orign", 'Twitter',
                                      'train.raw')
        test_filepath = os.path.join(base_data_dir,
                                      'Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines',
                                      "data_orign", 'Twitter',
                                      'test.raw')
        data_type_and_filepath = {
            'train': train_filepath,
            'dev': None,
            'test': test_filepath
        }
        data_type_and_datas = {}
        for data_type, filepath in data_type_and_filepath.items():
            if filepath is None:
                data_type_and_datas[data_type] = None
                continue
            lines = file_utils.read_all_lines(filepath)
            sentence = None
            term = None
            sentence_aspect_terms = collections.defaultdict(list)
            for i, line in enumerate(lines):
                if i % 3 == 0:
                    sentence = line
                elif i % 3 == 1:
                    term = line
                else:
                    label = int(line)
                    if label == 1:
                        polarity = 'positive'
                    elif label == 0:
                        polarity = 'neutral'
                    else:
                        polarity = 'negative'
                    sentence_complete = sentence.replace('$T$', term)
                    from_index = sentence.index('$T$')
                    to_index = from_index + len(term)
                    aspect_term = AspectTerm(term, polarity, from_index, to_index)
                    sentence_aspect_terms[sentence_complete].append(aspect_term)
            sentences = []
            for text, aspect_terms in sentence_aspect_terms.items():
                if len(aspect_terms) > 1:
                    print(text)
                sentence = AbsaSentence(text, None, None, aspect_terms)
                sentences.append(sentence)
            documents = [AbsaDocument(sentence.text, None, None, None, [sentence]) for sentence in sentences]
            data_type_and_datas[data_type] = documents
        train_data = data_type_and_datas['train']
        dev_data = data_type_and_datas['dev']
        test_data = data_type_and_datas['test']
        return train_data, dev_data, test_data


class Semeval2014Task4RestHard(Semeval2014Task4):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'conceptnet_augment_data.pkl')

    def _load_train_dev_test_data_by_filepath(self, train_filepath, test_filepath):
        data_type_and_filepath = {'train': train_filepath,
                                  'test': test_filepath}
        data_type_and_data = {}
        for data_type, filepath in data_type_and_filepath.items():
            lines = file_utils.read_all_lines(filepath)
            sentence_and_labels = defaultdict(list)
            for i in range(len(lines)):
                if i == 0:
                    continue
                line = lines[i]
                parts = line.split('\t')
                sentence_and_labels[parts[1]].append(parts[2:])

            sentences = []
            for sentence, labels in sentence_and_labels.items():
                aspect_categories = []
                for label in labels:
                    category = label[0]
                    if category == 'misc':
                        category = 'anecdotes/miscellaneous'
                    polarity = label[1]
                    if polarity == 'conflict':
                        continue
                    aspect_category = AspectCategory(category, polarity)
                    aspect_categories.append(aspect_category)
                if len(aspect_categories) < 2:
                    continue

                sentence = AbsaSentence(sentence, None, aspect_categories, None)
                sentences.append(sentence)
            documents = [AbsaDocument(sentence.text, None, None, None, [sentence]) for sentence in sentences]
            data_type_and_data[data_type] = documents
        return data_type_and_data['train'], None, data_type_and_data['test']

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST-HARD',
                                      'train.csv')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST-HARD',
                                     'test_public_gold.csv')
        return self._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class SemEval141516LargeRestHARD(Semeval2014Task4RestHard):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'SemEval-2014-Task-4-REST', 'origin',
                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines",
                                      'conceptnet_augment_data.pkl')

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-141516-LARGE-REST-HARD',
                                      'train.csv')
        test_filepath = os.path.join(base_data_dir, 'SemEval-141516-LARGE-REST-HARD',
                                     'test_public_gold.csv')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class Semeval2014Task4RestGCAE(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data_by_filepath(self, train_filepath, test_filepath):
        data_type_and_filepath = {'train': train_filepath,
                                  'test': test_filepath}
        data_type_and_data = {}
        for data_type, filepath in data_type_and_filepath.items():
            with open(filepath) as json_file:
                data_json = json.load(json_file)
            lines = []
            for i, element in enumerate(data_json):
                sentence_id = str(i)
                text = element['sentence']
                category = element['aspect']
                sentiment = element['sentiment']
                example = '\t'.join([sentence_id, text, category, sentiment])
                lines.append(example)

            sentence_and_labels = defaultdict(list)
            for i in range(len(lines)):
                if i == 0:
                    continue
                line = lines[i]
                parts = line.split('\t')
                sentence_and_labels[parts[1]].append(parts[2:])

            sentences = []
            for sentence, labels in sentence_and_labels.items():
                aspect_categories = []
                for label in labels:
                    category = label[0]
                    if category == 'misc':
                        category = 'anecdotes/miscellaneous'
                    polarity = label[1]
                    if polarity == 'conflict':
                        continue
                    aspect_category = AspectCategory(category, polarity)
                    aspect_categories.append(aspect_category)

                sentence = AbsaSentence(sentence, None, aspect_categories, None)
                sentences.append(sentence)
            documents = [AbsaDocument(sentence.text, None, None, None, [sentence]) for sentence in sentences]
            data_type_and_data[data_type] = documents
        return data_type_and_data['train'], None, data_type_and_data['test']

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-2014',
                                      'acsa_train.json')
        test_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-2014',
                                     'acsa_test.json')
        return self._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size, random_state=1234)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2014Task4RestHardGCAE(Semeval2014Task4RestGCAE):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-2014',
                                      'acsa_hard_train.json')
        test_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-2014',
                                     'acsa_hard_test.json')
        return self._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class SemEval141516LargeRestGCAE(Semeval2014Task4RestGCAE):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-large',
                                      'acsa_train.json')
        test_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-large',
                                     'acsa_test.json')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class SemEval141516LargeRest(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        rest14 = Semeval2014Task4Rest()
        train_dev_test_data, distinct_categories, distinct_polarities = \
            rest14.generate_acd_and_sc_data(dev_size=dev_size)
        distinct_categories = set(distinct_categories)
        distinct_polarities = set(distinct_polarities)

        rest15 = Semeval2015Task12Rest()
        rest15_train_dev_test_data, rest15_distinct_categories, rest15_distinct_polarities = \
            rest15.generate_acd_and_sc_data(dev_size=dev_size)

        rest16 = Semeval2016Task5RestSub1()
        rest16_train_dev_test_data, rest16_distinct_categories, rest16_distinct_polarities = \
            rest16.generate_acd_and_sc_data(dev_size=dev_size)

        rest1516_train_dev_test_data = {
            'train': rest15_train_dev_test_data['train'],
            'dev': rest15_train_dev_test_data['dev'],
            'test': rest15_train_dev_test_data['test'] + rest16_train_dev_test_data['test']
        }

        category_mapping = {
            'AMBIENCE#GENERAL': 'ambience',
            'DRINKS#PRICES': 'price',
            'DRINKS#QUALITY': 'drinks',
            'DRINKS#STYLE_OPTIONS': 'drinks',
            'FOOD#GENERAL': 'food',
            'FOOD#PRICES': 'price',
            'FOOD#QUALITY': 'food',
            'FOOD#STYLE_OPTIONS': 'food',
            'LOCATION#GENERAL': 'location',
            'RESTAURANT#GENERAL': 'restaurant',
            'RESTAURANT#MISCELLANEOUS': 'anecdotes/miscellaneous',
            'RESTAURANT#PRICES': 'price',
            'SERVICE#GENERAL': 'service'
        }
        for data_type, data in rest1516_train_dev_test_data.items():
            for sample in data:
                sentence = sample[0]
                labels = sample[1]
                if len(labels) == 0:
                    continue
                label_news = []
                aspect_categories_temp = {}
                for category, polarity in labels:
                    category = category_mapping[category]
                    if category not in aspect_categories_temp:
                        aspect_categories_temp[category] = set()
                    aspect_categories_temp[category].add(polarity)
                for category, polarities in aspect_categories_temp.items():
                    if len(polarities) == 1:
                        label_news.append((category, polarities.pop()))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    else:
                        if ('positive' in polarities and 'negative' in polarities) or 'conflict' in polarities:
                            label_news.append((category, 'conflict'))
                            distinct_categories.add(category)
                            distinct_polarities.add('conflict')
                        elif 'positive' in polarities and 'neutral' in polarities:
                            label_news.append((category, 'positive'))
                            distinct_categories.add(category)
                            distinct_polarities.add('positive')
                        else:
                            label_news.append((category, 'negative'))
                            distinct_categories.add(category)
                            distinct_polarities.add('negative')
                train_dev_test_data[data_type].append([sentence, label_news])

        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return train_dev_test_data, distinct_categories, distinct_polarities


class SemEval141516LargeRestWithRest14Categories(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        rest14 = Semeval2014Task4Rest()
        train_dev_test_data, distinct_categories, distinct_polarities = \
            rest14.generate_acd_and_sc_data(dev_size=dev_size)
        distinct_categories = set(distinct_categories)
        distinct_polarities = set(distinct_polarities)

        rest15 = Semeval2015Task12Rest()
        rest15_train_dev_test_data, rest15_distinct_categories, rest15_distinct_polarities = \
            rest15.generate_acd_and_sc_data(dev_size=dev_size)

        rest16 = Semeval2016Task5RestSub1()
        rest16_train_dev_test_data, rest16_distinct_categories, rest16_distinct_polarities = \
            rest16.generate_acd_and_sc_data(dev_size=dev_size)

        rest1516_train_dev_test_data = {
            'train': rest15_train_dev_test_data['train'],
            'dev': rest15_train_dev_test_data['dev'],
            'test': rest15_train_dev_test_data['test'] + rest16_train_dev_test_data['test']
        }

        category_mapping = {
            'AMBIENCE#GENERAL': 'ambience',
            'DRINKS#PRICES': 'price',
            'DRINKS#QUALITY': 'food',
            'DRINKS#STYLE_OPTIONS': 'food',
            'FOOD#GENERAL': 'food',
            #  rest14 case I thought the food isn't cheap at all compared to Chinatown.
            # <aspectCategory category="price" polarity="negative" />
            'FOOD#PRICES': 'price',
            'FOOD#QUALITY': 'food',
            'FOOD#STYLE_OPTIONS': 'food',
            'LOCATION#GENERAL': 'anecdotes/miscellaneous',
            'RESTAURANT#GENERAL': 'anecdotes/miscellaneous',
            'RESTAURANT#MISCELLANEOUS': 'anecdotes/miscellaneous',
            'RESTAURANT#PRICES': 'price',
            'SERVICE#GENERAL': 'service'
        }
        for data_type, data in rest1516_train_dev_test_data.items():
            for sample in data:
                sentence = sample[0]
                labels = sample[1]
                if len(labels) == 0:
                    continue
                label_news = []
                aspect_categories_temp = {}
                for category, polarity in labels:
                    category = category_mapping[category]
                    if category not in aspect_categories_temp:
                        aspect_categories_temp[category] = set()
                    aspect_categories_temp[category].add(polarity)
                for category, polarities in aspect_categories_temp.items():
                    if len(polarities) == 1:
                        label_news.append((category, polarities.pop()))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    else:
                        if ('positive' in polarities and 'negative' in polarities) or 'conflict' in polarities:
                            label_news.append((category, 'conflict'))
                            distinct_categories.add(category)
                            distinct_polarities.add('conflict')
                        elif 'positive' in polarities and 'neutral' in polarities:
                            label_news.append((category, 'positive'))
                            distinct_categories.add(category)
                            distinct_polarities.add('positive')
                        else:
                            label_news.append((category, 'negative'))
                            distinct_categories.add(category)
                            distinct_polarities.add('negative')
                train_dev_test_data[data_type].append([sentence, label_news])

        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return train_dev_test_data, distinct_categories, distinct_polarities


class SemEval1415LargeRestWithRest14Categories(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        rest14 = Semeval2014Task4Rest()
        train_dev_test_data, distinct_categories, distinct_polarities = \
            rest14.generate_acd_and_sc_data(dev_size=dev_size)
        distinct_categories = set(distinct_categories)
        distinct_polarities = set(distinct_polarities)

        rest15 = Semeval2015Task12Rest()
        rest15_train_dev_test_data, rest15_distinct_categories, rest15_distinct_polarities = \
            rest15.generate_acd_and_sc_data(dev_size=dev_size)

        rest1516_train_dev_test_data = {
            'train': rest15_train_dev_test_data['train'],
            'dev': rest15_train_dev_test_data['dev'],
            'test': rest15_train_dev_test_data['test']
        }

        category_mapping = {
            'AMBIENCE#GENERAL': 'ambience',
            'DRINKS#PRICES': 'price',
            'DRINKS#QUALITY': 'food',
            'DRINKS#STYLE_OPTIONS': 'food',
            'FOOD#GENERAL': 'food',
            'FOOD#PRICES': 'price',
            'FOOD#QUALITY': 'food',
            'FOOD#STYLE_OPTIONS': 'food',
            'LOCATION#GENERAL': 'anecdotes/miscellaneous',
            'RESTAURANT#GENERAL': 'anecdotes/miscellaneous',
            'RESTAURANT#MISCELLANEOUS': 'anecdotes/miscellaneous',
            'RESTAURANT#PRICES': 'price',
            'SERVICE#GENERAL': 'service'
        }
        for data_type, data in rest1516_train_dev_test_data.items():
            for sample in data:
                sentence = sample[0]
                labels = sample[1]
                if len(labels) == 0:
                    continue
                label_news = []
                aspect_categories_temp = {}
                for category, polarity in labels:
                    category = category_mapping[category]
                    if category not in aspect_categories_temp:
                        aspect_categories_temp[category] = set()
                    aspect_categories_temp[category].add(polarity)
                for category, polarities in aspect_categories_temp.items():
                    if len(polarities) == 1:
                        label_news.append((category, polarities.pop()))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    else:
                        if ('positive' in polarities and 'negative' in polarities) or 'conflict' in polarities:
                            label_news.append((category, 'conflict'))
                            distinct_categories.add(category)
                            distinct_polarities.add('conflict')
                        elif 'positive' in polarities and 'neutral' in polarities:
                            label_news.append((category, 'positive'))
                            distinct_categories.add(category)
                            distinct_polarities.add('positive')
                        else:
                            label_news.append((category, 'negative'))
                            distinct_categories.add(category)
                            distinct_polarities.add('negative')
                train_dev_test_data[data_type].append([sentence, label_news])

        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return train_dev_test_data, distinct_categories, distinct_polarities


class SemEval1416LargeRestWithRest14Categories(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        rest14 = Semeval2014Task4Rest()
        train_dev_test_data, distinct_categories, distinct_polarities = \
            rest14.generate_acd_and_sc_data(dev_size=dev_size)
        distinct_categories = set(distinct_categories)
        distinct_polarities = set(distinct_polarities)

        rest16 = Semeval2016Task5RestSub1()
        rest16_train_dev_test_data, rest16_distinct_categories, rest16_distinct_polarities = \
            rest16.generate_acd_and_sc_data(dev_size=dev_size)

        rest1516_train_dev_test_data = {
            'train': rest16_train_dev_test_data['train'],
            'dev': rest16_train_dev_test_data['dev'],
            'test': rest16_train_dev_test_data['test']
        }

        category_mapping = {
            'AMBIENCE#GENERAL': 'ambience',
            'DRINKS#PRICES': 'price',
            'DRINKS#QUALITY': 'food',
            'DRINKS#STYLE_OPTIONS': 'food',
            'FOOD#GENERAL': 'food',
            'FOOD#PRICES': 'price',
            'FOOD#QUALITY': 'food',
            'FOOD#STYLE_OPTIONS': 'food',
            'LOCATION#GENERAL': 'anecdotes/miscellaneous',
            'RESTAURANT#GENERAL': 'anecdotes/miscellaneous',
            'RESTAURANT#MISCELLANEOUS': 'anecdotes/miscellaneous',
            'RESTAURANT#PRICES': 'price',
            'SERVICE#GENERAL': 'service'
        }
        for data_type, data in rest1516_train_dev_test_data.items():
            for sample in data:
                sentence = sample[0]
                labels = sample[1]
                if len(labels) == 0:
                    continue
                label_news = []
                aspect_categories_temp = {}
                for category, polarity in labels:
                    category = category_mapping[category]
                    if category not in aspect_categories_temp:
                        aspect_categories_temp[category] = set()
                    aspect_categories_temp[category].add(polarity)
                for category, polarities in aspect_categories_temp.items():
                    if len(polarities) == 1:
                        label_news.append((category, polarities.pop()))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    else:
                        if ('positive' in polarities and 'negative' in polarities) or 'conflict' in polarities:
                            label_news.append((category, 'conflict'))
                            distinct_categories.add(category)
                            distinct_polarities.add('conflict')
                        elif 'positive' in polarities and 'neutral' in polarities:
                            label_news.append((category, 'positive'))
                            distinct_categories.add(category)
                            distinct_polarities.add('positive')
                        else:
                            label_news.append((category, 'negative'))
                            distinct_categories.add(category)
                            distinct_polarities.add('negative')
                train_dev_test_data[data_type].append([sentence, label_news])

        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return train_dev_test_data, distinct_categories, distinct_polarities


class SemEval141516LargeRestHARDGCAE(Semeval2014Task4RestGCAE):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-large',
                                      'acsa_hard_train.json')
        test_filepath = os.path.join(base_data_dir, 'GCAE', 'acsa-restaurant-large',
                                     'acsa_hard_test.json')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class MAMSATSA(Semeval2014Task4):
    """
    2019-emnlp-A_Challenge_Dataset_and_Effective_Models_for_Aspect_Based_Sentiment_Analysis
    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ATSA', 'raw',
                                                             "conceptnet_augment_data.pkl")

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ATSA', 'raw',
                                      "train.xml")
        test_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ATSA', 'raw',
                                     "test.xml")
        val_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ATSA', 'raw',
                                     "val.xml")
        return super()._load_semeval_by_filepath(train_filepath, test_filepath, val_filepath)


class SemEvalTripletData(BaseDataset):
    """
    Knowing What, How and Why: A Near Complete Solution for Aspect-based Sentiment Analysis
    """

    def __init__(self, configuration: dict={}):
        super().__init__(configuration)

    def _load_data_by_filepath(self, filepath):
        lines = file_utils.read_all_lines(filepath)
        sentences = []
        for line in lines:
            parts = line.split('####')
            text = parts[0]
            text_with_aspect_term = parts[1]
            word_labels = text_with_aspect_term.split(' ')

            start_index = 0
            last_label = 'O'
            aspect_term_index = -1
            aspect_term_parts = []
            aspect_term_polarity = ''
            aspect_terms = []
            for word_label in word_labels:
                word, label = word_label.split('=')
                polarity = ''
                if 'POS' in label:
                    polarity = 'positive'
                elif 'NEG' in label:
                    polarity = 'negative'
                elif 'NEU' in label:
                    polarity = 'neutral'

                # pass
                if label != last_label and len(aspect_term_parts) != 0:
                    term = ' '.join(aspect_term_parts)
                    aspect_term = AspectTerm(term, aspect_term_polarity, aspect_term_index,
                                             aspect_term_index + len(term))
                    aspect_terms.append(aspect_term)
                    aspect_term_index = -1
                    aspect_term_parts = []
                    aspect_term_polarity = ''

                # now
                if label.startswith('T'):
                    if len(aspect_term_parts) == 0:
                        aspect_term_index = start_index
                        aspect_term_polarity = polarity
                    aspect_term_parts.append(word)

                start_index = start_index + len(word) + 1
                last_label = label

            if len(aspect_term_parts) != 0:
                term = ' '.join(aspect_term_parts)
                aspect_term = AspectTerm(term, polarity, aspect_term_index,
                                         aspect_term_index + len(term))
                aspect_terms.append(aspect_term)

            sentence = AbsaSentence(text, None, None, aspect_terms)
            sentences.append(sentence)

        documents = [AbsaDocument(sentence.text, None, None, None, [sentence]) for sentence in sentences]
        return documents

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

        train_data = self._load_data_by_filepath(train_filepath)
        dev_data = self._load_data_by_filepath(dev_filepath)
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
        sentence_and_lines = defaultdict(list)
        for line in lines:
            original_line_data = json.loads(line)
            text = original_line_data['sentence']
            sentence_and_lines[text].append(original_line_data)

        sentences = []
        for text, original_line_datas in sentence_and_lines.items():
            first_original_line_data = original_line_datas[0]
            words = first_original_line_data['words']

            aspect_terms = []
            for original_line_data in original_line_datas:
                aspect_term_dict = original_line_data['aspect_term']
                polarity = original_line_data['polarity']
                if not self.is_include_this_sample(polarity):
                    continue

                term = aspect_term_dict['term']
                aspect_term_start_word_index = aspect_term_dict['start']
                if aspect_term_start_word_index == 0:
                    aspect_term_start_index = 0
                else:
                    aspect_term_start_index = len(' '.join(words[: aspect_term_start_word_index])) + 1
                aspect_term = AspectTerm(term, polarity, aspect_term_start_index,
                                         aspect_term_start_index + len(term))
                aspect_terms.append(aspect_term)
            if len(aspect_terms) == 0:
                continue

            sentence = AbsaSentence(text, None, None, aspect_terms)
            sentences.append(sentence)
        documents = [AbsaDocument(sentence.text, None, None, None, [sentence]) for sentence in sentences]
        return documents

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


class MAMSATSANlpcc2020(Semeval2014Task4):
    """
    2019-emnlp-A_Challenge_Dataset_and_Effective_Models_for_Aspect_Based_Sentiment_Analysis
    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)
        self.conceptnet_augment_data_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ATSA', 'raw',
                                                             "conceptnet_augment_data.pkl")

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'NLPCC-2020-Shared-Task-task2', 'Dataset_MAMS', 'ATSA',
                                      "train.xml")
        dev_filepath = os.path.join(base_data_dir, 'NLPCC-2020-Shared-Task-task2', 'Dataset_MAMS', 'ATSA',
                                     "dev.xml")
        test_filepath = os.path.join(base_data_dir, 'NLPCC-2020-Shared-Task-task2', 'Dataset_MAMS', 'ATSA',
                                    "test.xml")
        return super()._load_semeval_by_filepath(train_filepath, test_filepath, dev_filepath)


class MAMSACSA(Semeval2014Task4):
    """
    2019-emnlp-A_Challenge_Dataset_and_Effective_Models_for_Aspect_Based_Sentiment_Analysis
    """

    sentiment_path = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'sentiment_dict.json')

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ACSA', 'raw',
                                      "train.xml")
        test_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ACSA', 'raw',
                                     "test.xml")
        val_filepath = os.path.join(base_data_dir, 'MAMS-for-ABSA', 'MAMS-ACSA', 'raw',
                                     "val.xml")
        return super()._load_semeval_by_filepath(train_filepath, test_filepath, val_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)

        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class MAMSACSANlpcc2020(Semeval2014Task4):
    """
    2019-emnlp-A_Challenge_Dataset_and_Effective_Models_for_Aspect_Based_Sentiment_Analysis
    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'NLPCC-2020-Shared-Task-task2', 'Dataset_MAMS', 'ACSA',
                                      "train.xml")
        dev_filepath = os.path.join(base_data_dir, 'NLPCC-2020-Shared-Task-task2', 'Dataset_MAMS', 'ACSA',
                                     "dev.xml")
        test_filepath = os.path.join(base_data_dir, 'NLPCC-2020-Shared-Task-task2', 'Dataset_MAMS', 'ACSA',
                                    "test.xml")
        return super()._load_semeval_by_filepath(train_filepath, test_filepath, dev_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)

        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2015Task12(BaseDataset):
    """

    """

    def _load_train_dev_test_data_by_filepath(self, train_filepath, test_filepath):
        """

        :param train_filepath:
        :param test_filepath:
        :return:
        """
        datas = []
        for filepath in [train_filepath, test_filepath]:
            if filepath is None:
                datas.append(None)
                continue
            content = file_utils.read_all_content(filepath)
            soup = BeautifulSoup(content, "lxml")
            doc_tags = soup.find_all('review')
            docs = []
            for doc_tag in doc_tags:
                sentence_tags = doc_tag.find_all('sentence')
                doc_texts = []
                sentences = []
                for sentence_tag in sentence_tags:
                    text = sentence_tag.text
                    opinion_tags = sentence_tag.find_all('opinion')
                    aspect_terms = []
                    aspect_categories = []
                    for opinion_tag in opinion_tags:
                        category = opinion_tag['category']
                        polarity = opinion_tag['polarity']
                        if 'target' in opinion_tag.attrs:
                            term = opinion_tag['target']
                            from_index = opinion_tag['from']
                            to_index = opinion_tag['to']
                            aspect_term = AspectTerm(term, polarity, from_index, to_index, category)
                            aspect_terms.append(aspect_term)
                        else:
                            aspect_category = AspectCategory(category, polarity)
                            aspect_categories.append(aspect_category)
                    sentence = AbsaSentence(text, None, aspect_categories, aspect_terms)
                    sentences.append(sentence)
                    doc_texts.append(sentence.text)
                doc = AbsaDocument(''.join(doc_texts), None, None, None, sentences)
                docs.append(doc)
            datas.append(docs)
        train_data = datas[0]
        test_data = datas[1]
        dev_data = None
        return train_data, dev_data, test_data

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]+', ' ', sentence.text)
                    label = []
                    # entity#aspectaspect term：
                    # (1)，
                    # (2) ，
                    # (3) ，conflict
                    aspect_categories_temp = {}
                    for aspect_term in sentence.aspect_terms:
                        category = aspect_term.category
                        polarity = aspect_term.polarity
                        if category not in aspect_categories_temp:
                            aspect_categories_temp[category] = set()
                        aspect_categories_temp[category].add(polarity)
                    for category, polarities in aspect_categories_temp.items():
                        if len(polarities) == 1:
                            label.append((category, polarities.pop()))
                            distinct_categories.add(category)
                            distinct_polarities.add(polarity)
                        else:
                            if ('positive' in polarities and 'negative' in polarities) or 'conflict' in polarities:
                                label.append((category, 'conflict'))
                                distinct_categories.add(category)
                                distinct_polarities.add('conflict')
                            elif 'positive' in polarities and 'neutral' in polarities:
                                label.append((category, 'positive'))
                                distinct_categories.add(category)
                                distinct_polarities.add('positive')
                            else:
                                label.append((category, 'negative'))
                                distinct_categories.add(category)
                                distinct_polarities.add('negative')
                    if len(label) == 0:
                        continue
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2016Task5Sub1(Semeval2015Task12):
    """
    Semeval2016Task5Sub1
    """

    def generate_aspect_category_detection_data(self, test_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        for data_type, data in self.get_data_type_and_data_dict().items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        label.append(category)
                        distinct_categories.add(category)
                    samples.append([content, label])
            result[data_type] = samples
        if result['dev'] is None:
            original_train_samples = result['train']
            train_samples, dev_samples = train_test_split(original_train_samples, test_size=test_size)
            result['train'] = train_samples
            result['dev'] = dev_samples
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        return result, distinct_categories


class Semeval2016Task5Sub2(BaseDataset):
    """

    """

    def _load_train_dev_test_data_by_filepath(self, train_filepath, test_filepath):
        """

        :param train_filepath:
        :param test_filepath:
        :return:
        """
        datas = []
        for filepath in [train_filepath, test_filepath]:
            if filepath is None:
                datas.append(None)
                continue
            content = file_utils.read_all_content(filepath)
            soup = BeautifulSoup(content, "lxml")
            doc_tags = soup.find_all('review')
            docs = []
            for doc_tag in doc_tags:
                sentence_tags = doc_tag.find_all('sentence')
                doc_texts = []
                sentences = []
                for sentence_tag in sentence_tags:
                    text = sentence_tag.text
                    sentence = AbsaSentence(text, None, None, None)
                    sentences.append(sentence)
                    doc_texts.append(sentence.text)

                opinion_tags = doc_tag.find_all('opinion')
                aspect_terms = []
                aspect_categories = []
                for opinion_tag in opinion_tags:
                    category = opinion_tag['category']
                    polarity = opinion_tag['polarity']
                    if 'target' in opinion_tag:
                        term = opinion_tag['target']
                        from_index = opinion_tag['from']
                        to_index = opinion_tag['to']
                        aspect_term = AspectTerm(term, polarity, from_index, to_index, category)
                        aspect_terms.append(aspect_term)
                    else:
                        aspect_category = AspectCategory(category, polarity)
                        aspect_categories.append(aspect_category)
                doc = AbsaDocument(''.join(doc_texts), None, aspect_categories, aspect_terms, sentences)
                docs.append(doc)
            datas.append(docs)
        train_data = datas[0]
        test_data = datas[1]
        dev_data = None
        return train_data, dev_data, test_data

    def generate_acd_and_sc_data(self, dev_size=0.2, random_state=1234):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                content = re.sub('[\r\n]', ' ', document.text)
                label = []
                for aspect_category in document.aspect_categories:
                    category = aspect_category.category
                    polarity = aspect_category.polarity
                    label.append((category, polarity))
                    distinct_categories.add(category)
                    distinct_polarities.add(polarity)
                samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size, random_state=random_state)
        for data_type, data in result.items():
            category_distribution = {}
            for sample in data:
                sample_labels = [e[0] for e in sample[1]]
                for sample_label in sample_labels:
                    if sample_label not in category_distribution:
                        category_distribution[sample_label] = 0
                    category_distribution[sample_label] += 1
            category_distribution = list(category_distribution.items())
            category_distribution.sort(key=lambda x: x[0])
            logger.info('%s: %s' % (data_type, str(category_distribution)))
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities

    def generate_aspect_category_detection_data(self, test_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        for data_type, data in self.get_data_type_and_data_dict().items():
            if data is None:
                continue
            samples = []
            for document in data:
                content = re.sub('[\r\n]', ' ', document.text)
                label = []
                for aspect_category in document.aspect_categories:
                    category = aspect_category.category
                    label.append(category)
                    distinct_categories.add(category)
                samples.append([content, label])
            result[data_type] = samples
        if result['dev'] is None:
            original_train_samples = result['train']
            train_samples, dev_samples = train_test_split(original_train_samples, test_size=test_size)
            result['train'] = train_samples
            result['dev'] = dev_samples
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        return result, distinct_categories


class Semeval2015Task12Rest(Semeval2015Task12):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2015-Task-12-REST', 'origin',
                                      "ABSA15_RestaurantsTrain",
                                      'ABSA-15_Restaurants_Train_Final.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2015-Task-12-REST', 'origin',
                                     'ABSA15_Restaurants_Test.xml')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class Semeval2015Task12Lapt(Semeval2015Task12):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2015-Task-12-LAPT', 'origin',
                                      "ABSA15_LaptopsTrain",
                                      'ABSA-15_Laptops_Train_Data.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2015-Task-12-LAPT', 'origin',
                                     'ABSA15_Laptops_Test.xml')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]+', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    if len(label) == 0:
                        continue
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2015Task12Hotel(Semeval2015Task12):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        test_filepath = os.path.join(base_data_dir, 'SemEval-2015-Task-12-HOTEL', 'origin',
                                     'ABSA15_Hotels_Test.xml')
        return super()._load_train_dev_test_data_by_filepath(None, test_filepath)


class Semeval2016Task5ChCameSub1(Semeval2016Task5Sub1):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-CH-CAME-SB1', 'origin',
                                      "camera_corpus",
                                      'camera_training.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-CH-CAME-SB1', 'origin',
                                     'CH_CAME_SB1_TEST_.xml')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]+', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    if len(label) == 0:
                        continue
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2016Task5ChPhnsSub1(Semeval2016Task5Sub1):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-CH-PHNS-SB1', 'origin',
                                      'Chinese_phones_training.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-CH-PHNS-SB1', 'origin',
                                     'CH_PHNS_SB1_TEST_.xml')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]+', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    if len(label) == 0:
                        continue
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2016Task5LaptSub1(Semeval2016Task5Sub1):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-LAPT-SB1', 'origin',
                                      'ABSA16_Laptops_Train_SB1_v2.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-LAPT-SB1', 'origin',
                                     'EN_LAPT_SB1_TEST_.xml.gold')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)

    def generate_acd_and_sc_data(self, dev_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        distinct_polarities = set()
        data_type_and_data = self.get_data_type_and_data_dict()
        for data_type, data in data_type_and_data.items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]+', ' ', sentence.text)
                    label = []
                    for aspect_category in sentence.aspect_categories:
                        category = aspect_category.category
                        polarity = aspect_category.polarity
                        label.append((category, polarity))
                        distinct_categories.add(category)
                        distinct_polarities.add(polarity)
                    if len(label) == 0:
                        continue
                    samples.append([content, label])
            result[data_type] = samples
        super().generate_dev_data(result, dev_size)
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        distinct_polarities = list(distinct_polarities)
        distinct_polarities.sort()
        return result, distinct_categories, distinct_polarities


class Semeval2016Task5LaptSub2(Semeval2016Task5Sub2):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-LAPT-SB2', 'origin',
                                      'ABSA16_Laptops_Train_English_SB2.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-LAPT-SB2', 'origin',
                                     'EN_LAPT_SB2_TEST.xml.gold')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


class Semeval2016Task5RestSub1(Semeval2016Task5Sub1):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-REST-SB1', 'origin',
                                      'ABSA16_Restaurants_Train_SB1_v2.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-REST-SB1', 'origin',
                                     'EN_REST_SB1_TEST.xml.gold')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)

    def generate_aspect_category_detection_data(self, test_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        for data_type, data in self.get_data_type_and_data_dict().items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = set()
                    for aspect_term in sentence.aspect_terms:
                        category = aspect_term.category
                        label.add(category)
                        distinct_categories.add(category)
                    samples.append([content, list(label)])
            result[data_type] = samples
        if result['dev'] is None:
            original_train_samples = result['train']
            train_samples, dev_samples = train_test_split(original_train_samples, test_size=test_size)
            result['train'] = train_samples
            result['dev'] = dev_samples
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        return result, distinct_categories

    def generate_entity_detection_data(self, test_size=0.2):
        """

        :return:
        """
        result = {
            'train': None,
            'dev': None,
            'test': None
        }
        distinct_categories = set()
        for data_type, data in self.get_data_type_and_data_dict().items():
            if data is None:
                continue
            samples = []
            for document in data:
                for sentence in document.absa_sentences:
                    content = re.sub('[\r\n]', ' ', sentence.text)
                    label = set()
                    for aspect_term in sentence.aspect_terms:
                        category = aspect_term.category.split('#')[0]
                        label.add(category)
                        distinct_categories.add(category)
                    samples.append([content, list(label)])
            result[data_type] = samples
        if result['dev'] is None:
            original_train_samples = result['train']
            train_samples, dev_samples = train_test_split(original_train_samples, test_size=test_size)
            result['train'] = train_samples
            result['dev'] = dev_samples
        distinct_categories = list(distinct_categories)
        distinct_categories.sort()
        return result, distinct_categories


class Semeval2016Task5RestSub2(Semeval2016Task5Sub2):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-REST-SB2', 'origin',
                                      'ABSA16_Restaurants_Train_English_SB2.xml')
        test_filepath = os.path.join(base_data_dir, 'SemEval-2016-Task-5-REST-SB2', 'origin',
                                     'EN_REST_SB2_TEST.xml.gold')
        return super()._load_train_dev_test_data_by_filepath(train_filepath, test_filepath)


def load_csv_data(filepath, skip_first_line=True):
    """

    :param filepath:
    :param skip_first_line:
    :return:
    """
    result = []
    lines = file_utils.read_all_lines(filepath)
    for line in lines:
        rows = csv.reader([line])
        for row in rows:
            result.append(row)
            if len(row) != len(result[0]):
                print(row)
    if skip_first_line:
        result = result[1:]
    return result


class Bdci2019InternetNews(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        train_filepath = os.path.join(base_data_dir, 'bdci2019', '',
                                      'Train_DataSet.csv')
        train_label_filepath = os.path.join(base_data_dir, 'bdci2019', '',
                                      'Train_DataSet_Label.csv')
        train_rows = load_csv_data(train_filepath)
        train_label_rows = load_csv_data(train_label_filepath)

        test_filepath = os.path.join(base_data_dir, 'bdci2019', '',
                                      'Test_DataSet.csv')
        test_label_filepath = os.path.join(base_data_dir, 'bdci2019', '',
                                      'submit_example.csv')
        test_rows = load_csv_data(test_filepath)
        test_label_rows = load_csv_data(test_label_filepath)

        data = {
            'train': [train_rows, train_label_rows],
            'test': [test_rows, test_label_rows]
        }
        result = {}
        for data_type, [data_rows, label_rows] in data.items():
            samples = []
            for i in range(len(data_rows)):
                sample_id = data_rows[i][0]
                data = '。'.join(data_rows[i][1:])
                label = label_rows[i][1]
                sample = Text(data, label, sample_id=sample_id)
                samples.append(sample)
            result[data_type] = samples
        train_data = result['train']
        dev_data = None
        test_data = result['test']
        return train_data, dev_data, test_data


class Bdci2019FinancialNegative(BaseDataset):
    """

    """

    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _load_train_dev_test_data(self):
        """

        :return:
        """
        train_filepath = os.path.join(base_data_dir, 'bdci2019', '',
                                      'Train_Data.csv')
        train_rows = load_csv_data(train_filepath)
        test_filepath = os.path.join(base_data_dir, 'bdci2019', '',
                                     'Test_Data.csv')
        test_label_filepath = os.path.join(base_data_dir, 'bdci2019', '',
                                           'Submit_Example.csv')
        test_rows = load_csv_data(test_filepath)
        test_label_rows = load_csv_data(test_label_filepath)
        data_rows = {
            'train': train_rows,
            'test': test_rows
        }
        data = {
            'train': None,
            'test': None
        }
        for data_type, rows in data_rows.items():
            samples = []
            for row in rows:
                sample_id = row[0]
                title = row[1]
                content = row[2]
                entities = row[3].split(';')
                text_polarity = row[4] if data_type == 'train' else 0
                key_entities = row[5].split(';') if data_type == 'train' else []
                aspect_categories = []
                for entity in entities:
                    if entity in key_entities:
                        polarity = '1'
                    else:
                        polarity = '0'
                    aspect_category = AspectCategory(entity, polarity)
                    aspect_categories.append(aspect_category)
                text = '%scontent-begin。%s' % (title, content)
                document = AbsaDocument(text, text_polarity, aspect_categories, None, None, sample_id=sample_id)
                samples.append(document)
            data[data_type] = samples
        train_data = data['train']
        dev_data = None
        test_data = data['test']
        return train_data, dev_data, test_data


suported_dataset_names_and_data_loader = {
    'SemEval-2014-Task-4-LAPT': Semeval2014Task4Lapt,
    'SemEval-2014-Task-4-REST': Semeval2014Task4Rest,
    'TWITTER': TWITTER,
    'SemEval-2014-Task-4-REST-DevSplits': Semeval2014Task4RestDevSplits,
    'SemEval-2014-Task-4-REST-DevSplits-Aspect-Term': Semeval2014Task4RestDevSplitsAspectTerm,
    'SemEval-2014-Task-4-LAPT-DevSplits-Aspect-Term': Semeval2014Task4LaptDevSplitsAspectTerm,
    'SemEval-2014-Task-4-REST-Hard': Semeval2014Task4RestHard,
    'Se1415Category-DevSplits': Se1415CategoryDevSplits,
    'SemEval-141516-LARGE-REST': SemEval141516LargeRest,
    'SemEval-141516-LARGE-REST-WithRest14Categories': SemEval141516LargeRestWithRest14Categories,
    'SemEval-1415-LARGE-REST-WithRest14Categories': SemEval1415LargeRestWithRest14Categories,
    'SemEval-1416-LARGE-REST-WithRest14Categories': SemEval1416LargeRestWithRest14Categories,
    'SemEval-141516-LARGE-REST-HARD': SemEval141516LargeRestHARD,
    'SemEval-2014-Task-4-REST-GCAE': Semeval2014Task4RestGCAE,
    'SemEval-2014-Task-4-REST-HARD-GCAE': Semeval2014Task4RestHardGCAE,
    'SemEval-141516-LARGE-REST-GCAE': SemEval141516LargeRestGCAE,
    'SemEval-141516-LARGE-REST-HARD-GCAE': SemEval141516LargeRestHARDGCAE,
    'SemEval-2015-Task-12-LAPT': Semeval2015Task12Lapt,
    'SemEval-2015-Task-12-REST': Semeval2015Task12Rest,
    'SemEval-2015-Task-12-HOTEL': Semeval2015Task12Hotel,
    'SemEval-2016-Task-5-CH-CAME-SB1': Semeval2016Task5ChCameSub1,
    'SemEval-2016-Task-5-CH-PHNS-SB1': Semeval2016Task5ChPhnsSub1,
    'SemEval-2016-Task-5-LAPT-SB1': Semeval2016Task5LaptSub1,
    'SemEval-2016-Task-5-LAPT-SB2': Semeval2016Task5LaptSub2,
    'SemEval-2016-Task-5-REST-SB1': Semeval2016Task5RestSub1,
    'SemEval-2016-Task-5-REST-SB2': Semeval2016Task5RestSub2,
    'bdci2019-internet-news-sa': Bdci2019InternetNews,
    'bdci2019-financial-negative': Bdci2019FinancialNegative,
    'AsgcnData2014Lapt': AsgcnData2014Lapt,
    'AsgcnData2014Rest': AsgcnData2014Rest,
    'AsgcnData2015Rest': AsgcnData2015Rest,
    'AsgcnData2016Rest': AsgcnData2016Rest,
    'MAMSACSA': MAMSACSA,
    'MAMSATSA': MAMSATSA,
    'MAMSACSANlpcc2020': MAMSACSANlpcc2020,
    'MAMSATSANlpcc2020': MAMSATSANlpcc2020,
    'yelp-dataset': YelpDataset,
    'nlpcc2012-weibo-sa': Nlpcc2012WeiboSa,
    'feed_comment': FeedComment,
    'MR': None,
    'SST-1': None,
    'SST-2': None,
    'CR': None,
    'triplet_rest14': TripletRest14,
    'triplet_lapt14': TripletLapt14,
    'triplet_rest15': TripletRest15,
    'triplet_rest16': TripletRest16,
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


if __name__ == '__main__':
    dataset_name = 'ASMOTEDataRest14'
    dataset = get_dataset_class_by_name(dataset_name)()
    data_type_and_mm_num = collections.defaultdict(int)
    for data_type, data in dataset.get_data_type_and_data_dict().items():
        for doc in data:
            data_type_and_mm_num['total_%s' % data_type] += 1
            aspect_terms = doc.absa_sentences[0].aspect_terms
            distinct_polarities = set()
            for aspect_term in aspect_terms:
                distinct_polarities.add(aspect_term.polarity)
            if len(distinct_polarities) > 1:
                data_type_and_mm_num[data_type] += 1
    print(data_type_and_mm_num)








