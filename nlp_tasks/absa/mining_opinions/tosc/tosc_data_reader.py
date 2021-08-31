# -*- coding: utf-8 -*-


import copy

from typing import *
from overrides import overrides
import pickle
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField, LabelField, MultiLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
import torch.nn.functional as F
from allennlp.training import metrics
from allennlp.models import BasicClassifier
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')
english_stop_words.extend([',', '.', '?', ';', '-', ':', '\'', '"', '(', ')', '!'])

from nlp_tasks.utils import corenlp_factory
from nlp_tasks.utils import create_graph
from nlp_tasks.utils import my_corenlp
from nlp_tasks.absa.data_adapter import data_object


class TextAspectInSentimentOut(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        start_index = 0
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        sample['words'] = words
        for i in range(len(word_indices_of_aspect_terms)):
            start_index = word_indices_of_aspect_terms[i][0]
            end_index = word_indices_of_aspect_terms[i][1]
            aspect_term_text = ' '.join(words[start_index: end_index])

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field
        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutBertForTOSC(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    def add_bert_words_and_word_index_bert_indices(self, words, fields, sample):
        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            if self.configuration['position_and_second_sentence']:
                aspect_term_dict = sample['aspect_terms'][0].metadata['opinion']['aspect_term']
                aspect_term_dict_start = aspect_term_dict['start']
                aspect_term_dict_end = aspect_term_dict['end']
                if aspect_term_dict_start <= i < aspect_term_dict_end:
                    word = 'aspect'
            bert_ws = self.bert_tokenizer.tokenize(word.lower())
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')

        if self.configuration['second_sentence']:
            words_of_aspect_term = sample['aspect_terms'][0].metadata['opinion']['aspect_term']['term']
            bert_words.extend(self.bert_tokenizer.tokenize(words_of_aspect_term.lower()))
            bert_words.append('[SEP]')

        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        fields['bert'] = bert_text_field
        sample['bert_words'] = bert_words
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        start_index = 0
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        sample['words'] = words
        for i in range(len(word_indices_of_aspect_terms)):
            start_index = word_indices_of_aspect_terms[i][0]
            end_index = word_indices_of_aspect_terms[i][1]
            aspect_term_text = ' '.join(words[start_index: end_index])

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        self.add_bert_words_and_word_index_bert_indices(words, fields, sample)

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field
        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutBertWithPositionForTOSC(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    def convert_to_relative_position(self, positions: List[int], aspect_positions: List[int]):
        """

        :param positions:
        :param aspect_positions:
        :return:
        """
        result = [index for index in range(len(positions))]
        for i in result:
            if i < aspect_positions[0]:
                result[i] = aspect_positions[0] - i
            elif i in aspect_positions:
                result[i] = 0
            else:
                result[i] = i - aspect_positions[-1]
        return result

    def add_bert_words_and_word_index_bert_indices(self, words, fields, sample):
        bert_words = ['[CLS]']
        aspect_indices = []
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            if self.configuration['position_and_second_sentence']:
                aspect_term_dict = sample['aspect_terms'][0].metadata['opinion']['aspect_term']
                aspect_term_dict_start = aspect_term_dict['start']
                aspect_term_dict_end = aspect_term_dict['end']
                if aspect_term_dict_start <= i < aspect_term_dict_end:
                    word = 'aspect'
            bert_ws = self.bert_tokenizer.tokenize(word.lower())
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)

                aspect_term_dict = sample['aspect_terms'][0].metadata['opinion']['aspect_term']
                aspect_term_dict_start = aspect_term_dict['start']
                aspect_term_dict_end = aspect_term_dict['end']
                if aspect_term_dict_start <= i < aspect_term_dict_end:
                    aspect_indices.append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        bert_position = list(range(len(bert_words)))

        if self.configuration['second_sentence']:
            words_of_aspect_term = sample['aspect_terms'][0].metadata['opinion']['aspect_term']['term']
            bert_words.extend(self.bert_tokenizer.tokenize(words_of_aspect_term.lower()))
            bert_position.extend(aspect_indices + [aspect_indices[-1]] * (len(self.bert_tokenizer.tokenize(words_of_aspect_term.lower())) - len(aspect_indices)))

            bert_words.append('[SEP]')
            bert_position.append(len(bert_words) - 1)

        if self.configuration['relative_position']:
            bert_position = self.convert_to_relative_position(bert_position, aspect_indices)

        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        fields['bert'] = bert_text_field

        bert_position_field = ArrayField(np.array(bert_position), padding_value=len(bert_words))
        fields['bert_position'] = bert_position_field

        sample['bert_words'] = bert_words
        sample['bert_position'] = bert_position
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        start_index = 0
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        sample['words'] = words
        for i in range(len(word_indices_of_aspect_terms)):
            start_index = word_indices_of_aspect_terms[i][0]
            end_index = word_indices_of_aspect_terms[i][1]
            aspect_term_text = ' '.join(words[start_index: end_index])

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        self.add_bert_words_and_word_index_bert_indices(words, fields, sample)

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field
        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutConsideringTarget(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    def _insert_special_token_into_text(self, words: List[str], start: int, end: int):
        """

        :param words:
        :param start:
        :param end:
        :return:
        """
        words.insert(start, '#')
        words.insert(end + 1, '$')

    def _add_special_token(self, sample: dict):
        """

        :param sample:
        :return:
        """
        sample['text_backup'] = copy.deepcopy(sample['text'])
        sample['aspect_terms_backup'] = copy.deepcopy(sample['aspect_terms'])

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        words: List[str] = text.split(' ')

        aspect = sample['aspect_terms'][0].metadata['opinion']['aspect_term']
        aspect_start = aspect['start']
        aspect_end = aspect['end']
        self._insert_special_token_into_text(words, aspect_start, aspect_end)
        sample['text'] = ' '.join(words)
        aspect_start_real = aspect_start + 1
        aspect_end_real = aspect_end + 1

        for aspect_term in sample['aspect_terms']:
            aspect_term.metadata['opinion']['aspect_term']['start_backup'] = aspect_term.metadata['opinion']['aspect_term']['start']
            aspect_term.metadata['opinion']['aspect_term']['end_backup'] = aspect_term.metadata['opinion']['aspect_term']['end']
            aspect_term.metadata['opinion']['opinion_term']['start_backup'] = aspect_term.metadata['opinion']['opinion_term']['start']
            aspect_term.metadata['opinion']['opinion_term']['end_backup'] = aspect_term.metadata['opinion']['opinion_term']['end']

            aspect_term.metadata['opinion']['aspect_term']['start'] = aspect_start_real
            aspect_term.metadata['opinion']['aspect_term']['end'] = aspect_end_real

            if aspect_term.metadata['opinion']['opinion_term']['start'] > aspect_start:
                aspect_term.metadata['opinion']['opinion_term']['start'] += 2
                aspect_term.metadata['opinion']['opinion_term']['end'] += 2
                aspect_term.from_index = len(' '.join(words[: aspect_term.metadata['opinion']['opinion_term']['start']])) + 1
                aspect_term.to_index = len(' '.join(words[: aspect_term.metadata['opinion']['opinion_term']['end']]))

    @overrides
    def text_to_instance(self, sample: dict) -> Instance:
        self._add_special_token(sample)

        fields = {}

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        start_index = 0
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        sample['words'] = words
        for i in range(len(word_indices_of_aspect_terms)):
            start_index = word_indices_of_aspect_terms[i][0]
            end_index = word_indices_of_aspect_terms[i][1]
            aspect_term_text = ' '.join(words[start_index: end_index])

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field
        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutConsideringTargetBertForTOSC(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None,
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    def add_bert_words_and_word_index_bert_indices(self, words, fields, sample):
        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word.lower())
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        fields['bert'] = bert_text_field
        sample['bert_words'] = bert_words
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

    def _insert_special_token_into_text(self, words: List[str], start: int, end: int):
        """

        :param words:
        :param start:
        :param end:
        :return:
        """
        words.insert(start, '#')
        words.insert(end + 1, '$')

    def _add_special_token(self, sample: dict):
        """

        :param sample:
        :return:
        """
        sample['text_backup'] = copy.deepcopy(sample['text'])
        sample['aspect_terms_backup'] = copy.deepcopy(sample['aspect_terms'])

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        words: List[str] = text.split(' ')

        aspect = sample['aspect_terms'][0].metadata['opinion']['aspect_term']
        aspect_start = aspect['start']
        aspect_end = aspect['end']
        self._insert_special_token_into_text(words, aspect_start, aspect_end)
        sample['text'] = ' '.join(words)
        aspect_start_real = aspect_start + 1
        aspect_end_real = aspect_end + 1

        for aspect_term in sample['aspect_terms']:
            aspect_term.metadata['opinion']['aspect_term']['start_backup'] = aspect_term.metadata['opinion']['aspect_term']['start']
            aspect_term.metadata['opinion']['aspect_term']['end_backup'] = aspect_term.metadata['opinion']['aspect_term']['end']
            aspect_term.metadata['opinion']['opinion_term']['start_backup'] = aspect_term.metadata['opinion']['opinion_term']['start']
            aspect_term.metadata['opinion']['opinion_term']['end_backup'] = aspect_term.metadata['opinion']['opinion_term']['end']

            aspect_term.metadata['opinion']['aspect_term']['start'] = aspect_start_real
            aspect_term.metadata['opinion']['aspect_term']['end'] = aspect_end_real

            if aspect_term.metadata['opinion']['opinion_term']['start'] > aspect_start:
                aspect_term.metadata['opinion']['opinion_term']['start'] += 2
                aspect_term.metadata['opinion']['opinion_term']['end'] += 2
                aspect_term.from_index = len(' '.join(words[: aspect_term.metadata['opinion']['opinion_term']['start']])) + 1
                aspect_term.to_index = len(' '.join(words[: aspect_term.metadata['opinion']['opinion_term']['end']]))

    @overrides
    def text_to_instance(self, sample: dict) -> Instance:
        self._add_special_token(sample)

        fields = {}

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        start_index = 0
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        sample['words'] = words
        for i in range(len(word_indices_of_aspect_terms)):
            start_index = word_indices_of_aspect_terms[i][0]
            end_index = word_indices_of_aspect_terms[i][1]
            aspect_term_text = ' '.join(words[start_index: end_index])

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        self.add_bert_words_and_word_index_bert_indices(words, fields, sample)

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field
        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for sample in samples:
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutForBert(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            if self.configuration['model_name'] == 'aspect-term-aware-Bert':
                pieces.append('#')
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            if self.configuration['model_name'] == 'aspect-term-aware-Bert':
                pieces.append('#')
                piece_labels.append(0)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.lower().strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        start_index = 1
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        words = ['[CLS]'] + words + ['[SEP]']
        sample['words'] = words
        for i in range(len(word_indices_of_aspect_terms)):
            start_index = word_indices_of_aspect_terms[i][0]
            end_index = word_indices_of_aspect_terms[i][1]
            aspect_term_text = ' '.join(words[start_index: end_index])
            if labels[i].term != aspect_term_text:
                print('real_term: %s generated_term: %s sentence: %s' % (labels[i].term, aspect_term_text, text))

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field
        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for i, sample in enumerate(samples):
                # if i != 0 and 'Opt for the spectacular Emperor\'s Meal' not in sample[0]:
                #     continue
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutForAtsaBert(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        labels: List[data_object.AspectTerm] = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            if self.configuration['aspect_term_aware']:
                pieces.append('#')
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            if self.configuration['aspect_term_aware']:
                if self.configuration['same_special_token']:
                    pieces.append('#')
                else:
                    pieces.append('$')
                piece_labels.append(0)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.lower().strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        # the first token is '[CLS]'
        start_index = 1
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        words = ['[CLS]'] + words + ['[SEP]']
        sample['words'] = words

        if self.configuration['pair']:
            assert self.configuration['sample_mode'] == 'single', 'While pair is True, sample_mode must be single'
            words_of_aspect_term = self.tokenizer(labels[0].term.lower().strip())
            words = words + words_of_aspect_term + ['[SEP]']

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field

        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for i, sample in enumerate(samples):
                # if i != 0 and 'Opt for the spectacular Emperor\'s Meal' not in sample[0]:
                #     continue
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutForAtsaLSTM(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, text):
        graph = create_graph.create_dependency_graph_for_dgl(text, self.spacy_nlp, None)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        labels: List[data_object.AspectTerm] = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            if self.configuration['aspect_term_aware']:
                pieces.append('#')
                piece_labels.append(0)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            if self.configuration['aspect_term_aware']:
                if self.configuration['same_special_token']:
                    pieces.append('#')
                else:
                    pieces.append('$')
                piece_labels.append(0)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        words_of_pieces = [self.tokenizer(piece.lower().strip()) for piece in pieces]
        word_indices_of_aspect_terms = []
        # the first token is '[CLS]'
        start_index = 0
        for i in range(len(words_of_pieces)):
            words_of_piece = words_of_pieces[i]
            end_index = start_index + len(words_of_piece)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        sample['words'] = words

        graph = self._build_graph(text)
        sample['graph'] = graph

        tokens = [Token(word) for word in words]
        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field

        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for i, sample in enumerate(samples):
                # if i != 0 and 'Opt for the spectacular Emperor\'s Meal' not in sample[0]:
                #     continue
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)


class TextAspectInSentimentOutForSyntaxAwareBert(DatasetReader):
    def __init__(self, polarities: List[str],
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.polarities = polarities
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration

    def _build_graph(self, word_and_word_pieces, node_num):
        graph = create_graph.create_dependency_graph_for_dgl_for_syntax_aware_atsa_bert(word_and_word_pieces,
                                                                                        self.spacy_nlp, node_num)
        return graph

    @overrides
    def text_to_instance(self, sample: list) -> Instance:
        fields = {}

        text: str = sample['text'].strip()
        labels = sample['aspect_terms']

        pieces = []
        piece_labels = []
        last_label_end_index = 0
        for i, label in enumerate(labels):
            if label.from_index != last_label_end_index:
                pieces.append(text[last_label_end_index: label.from_index])
                piece_labels.append(0)
            if self.configuration['model_name'] == 'aspect-term-aware-bert-syntax':
                pieces.append('#')
                piece_labels.append(2)
            pieces.append(text[label.from_index: label.to_index])
            piece_labels.append(1)
            if self.configuration['model_name'] == 'aspect-term-aware-bert-syntax':
                pieces.append('#')
                piece_labels.append(2)
            last_label_end_index = label.to_index
            if i == len(labels) - 1 and label.to_index != len(text):
                pieces.append(text[label.to_index:])
                piece_labels.append(0)

        # words_of_pieces = [self.tokenizer(piece.lower().strip()) for piece in pieces]
        words_of_pieces = []
        word_indices_of_aspect_terms = []
        start_index = 1
        # ，[word，word piecesindex，word piecesindex]
        word_and_word_pieces = []
        for i in range(len(pieces)):
            piece = pieces[i]
            doc = self.spacy_nlp(piece.lower().strip())
            words_of_piece = [word.text for word in doc]
            word_pieces_of_words = []
            word_pieces_start_index = start_index
            for word in words_of_piece:
                word_pieces_of_word = self.tokenizer(word.lower().strip())
                word_pieces_of_words.extend(word_pieces_of_word)
                word_pieces_end_index = word_pieces_start_index + len(word_pieces_of_word)
                if piece_labels[i] != 2:
                    word_and_word_pieces.append([word, word_pieces_start_index, word_pieces_end_index,
                                                 word_pieces_of_word])
                word_pieces_start_index = word_pieces_end_index
            words_of_pieces.append(word_pieces_of_words)

            end_index = start_index + len(word_pieces_of_words)
            if piece_labels[i] == 1:
                word_indices_of_aspect_terms.append([start_index, end_index])
            start_index = end_index
        sample['word_indices_of_aspect_terms'] = word_indices_of_aspect_terms

        words = []
        for words_of_piece in words_of_pieces:
            words.extend(words_of_piece)
        words = ['[CLS]'] + words + ['[SEP]']
        sample['words'] = words

        graph = self._build_graph(word_and_word_pieces, len(words))
        sample['graph'] = graph

        tokens = [Token(word) for word in words]

        sentence_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = sentence_field

        position = [Token(str(i)) for i in range(len(tokens))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field
        if self.configuration['sample_mode'] == 'single':
            max_aspect_term_num = 1
        else:
            max_aspect_term_num = self.configuration['max_aspect_term_num']
        polarity_labels = [-100] * max_aspect_term_num
        for i, aspect_term in enumerate(sample['aspect_terms']):
            polarity_labels[i] = self.polarities.index(aspect_term.polarity)
        label_field = ArrayField(np.array(polarity_labels))
        fields["label"] = label_field
        polarity_mask = [1 if polarity_labels[i] != -100 else 0 for i in range(max_aspect_term_num)]
        polarity_mask_field = ArrayField(np.array(polarity_mask))
        fields['polarity_mask'] = polarity_mask_field

        # stop_word_labels = [1 if word in english_stop_words else 0 for word in words]
        # stop_word_num = sum(stop_word_labels)
        # stop_word_labels = [label / stop_word_num for label in stop_word_labels]
        # sample.append(stop_word_labels)

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sample_mode = self.configuration['sample_mode']
        if sample_mode == 'single':
            for sample in samples:
                for label in sample[1]:
                    yield self.text_to_instance({'text': sample[0], 'aspect_terms': [label]})
        elif sample_mode == 'multi':
            for i, sample in enumerate(samples):
                # if i != 0 and 'Opt for the spectacular Emperor\'s Meal' not in sample[0]:
                #     continue
                yield self.text_to_instance({'text': sample[0], 'aspect_terms': sample[1]})
        else:
            raise NotImplementedError('sample model: %s' % sample_mode)