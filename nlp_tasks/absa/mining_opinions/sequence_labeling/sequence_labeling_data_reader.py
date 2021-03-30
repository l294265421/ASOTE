# -*- coding: utf-8 -*-


from collections import OrderedDict
import copy

from typing import *
from overrides import overrides
import pickle
import copy

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField, LabelField, MultiLabelField, SequenceLabelField
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
from nlp_tasks.utils.sentence_segmenter import BaseSentenceSegmenter, NltkSentenceSegmenter


class DatasetReaderForTCBiLSTM(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        words = sample['words']
        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        position = [Token(str(i)) for i in range(len(words))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags = sample['opinion_words_tags']
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        target_tags = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        sample['target_start_index'] = target_start_index
        sample['target_end_index'] = target_end_index

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForTermBiLSTM(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        words.insert(target_start_index, '#')
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForTermBiLSTMForMFGData(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        sample['target_tags_backup'] = copy.deepcopy(sample['target_tags'])
        sample['opinion_words_tags_backup'] = copy.deepcopy(sample['opinion_words_tags'])
        sample['words_backup'] = copy.deepcopy(sample['words'])

        sample['opinion_words_tags'] = [e[-1] for e in sample['opinion_words_tags']]

        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        words.insert(target_start_index, '#')
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForAsoTermBiLSTM(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        sample['words_backup'] = sample['words']
        sample['words'] = copy.deepcopy(sample['words'])
        sample['target_tags_backup'] = sample['target_tags']
        sample['target_tags'] = copy.deepcopy(sample['target_tags'])
        sample['opinion_words_tags_backup'] = sample['opinion_words_tags']
        sample['opinion_words_tags'] = copy.deepcopy(sample['opinion_words_tags'])

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        words.insert(target_start_index, '#')
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [real_target_start_index + 1, real_target_end_index]
        # if 'polarity' in sample:
        #     polarity_index = self.polarities.index(sample['polarity'])
        #     polarity_label_field = LabelField(polarity_index, skip_indexing=True,
        #                                       label_namespace='polarity_labels')
        #     fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForAsoTermBert(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}

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

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        sample['words_backup'] = sample['words']
        sample['words'] = copy.deepcopy(sample['words'])
        sample['target_tags_backup'] = sample['target_tags']
        sample['target_tags'] = copy.deepcopy(sample['target_tags'])
        sample['opinion_words_tags_backup'] = sample['opinion_words_tags']
        sample['opinion_words_tags'] = copy.deepcopy(sample['opinion_words_tags'])

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        words.insert(target_start_index, '#')
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        self.add_bert_words_and_word_index_bert_indices(words, fields, sample)

        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [real_target_start_index + 1, real_target_end_index]
        # if 'polarity' in sample:
        #     polarity_index = self.polarities.index(sample['polarity'])
        #     polarity_label_field = LabelField(polarity_index, skip_indexing=True,
        #                                       label_namespace='polarity_labels')
        #     fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForMilAsoTermBiLSTM(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        sample['words_backup'] = sample['words']
        sample['words'] = copy.deepcopy(sample['words'])
        sample['target_tags_backup'] = sample['target_tags']
        sample['target_tags'] = copy.deepcopy(sample['target_tags'])
        sample['opinion_words_tags_backup'] = sample['opinion_words_tags']
        sample['opinion_words_tags'] = copy.deepcopy(sample['opinion_words_tags'])

        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        words.insert(target_start_index, '#')
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List[str] = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags_with_polarity'] = tags

            tags = [tag if '-' not in tag else tag[tag.index('-') + 1:] for tag in tags]
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [real_target_start_index + 1, real_target_end_index]
        if 'polarity' in sample:
            # polarity_index = self.polarities.index(sample['polarity'])
            # polarity_label_field = LabelField(polarity_index, skip_indexing=True,
            #                                   label_namespace='polarity_labels')
            polarity_indices = []
            if sample['polarity'] == 'conflict':
                polarity_indices.append(self.polarities.index('positive'))
                polarity_indices.append(self.polarities.index('negative'))
            else:
                polarity_indices.append(self.polarities.index(sample['polarity']))
            polarity_label_field = MultiLabelField(polarity_indices,
                                                   label_namespace='polarity_labels',
                                                   skip_indexing=True,
                                                   num_labels=len(self.polarities))
            fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForMilAsoTermBert(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}

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

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        sample['words_backup'] = sample['words']
        sample['words'] = copy.deepcopy(sample['words'])
        sample['target_tags_backup'] = sample['target_tags']
        sample['target_tags'] = copy.deepcopy(sample['target_tags'])
        sample['opinion_words_tags_backup'] = sample['opinion_words_tags']
        sample['opinion_words_tags'] = copy.deepcopy(sample['opinion_words_tags'])

        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        words.insert(target_start_index, '#')
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        self.add_bert_words_and_word_index_bert_indices(words, fields, sample)

        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List[str] = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags_with_polarity'] = tags

            tags = [tag if '-' not in tag else tag[tag.index('-') + 1:] for tag in tags]
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [real_target_start_index + 1, real_target_end_index]
        if 'polarity' in sample:
            # polarity_index = self.polarities.index(sample['polarity'])
            # polarity_label_field = LabelField(polarity_index, skip_indexing=True,
            #                                   label_namespace='polarity_labels')
            polarity_indices = []
            if sample['polarity'] == 'conflict':
                polarity_indices.append(self.polarities.index('positive'))
                polarity_indices.append(self.polarities.index('negative'))
            else:
                polarity_indices.append(self.polarities.index(sample['polarity']))
            polarity_label_field = MultiLabelField(polarity_indices,
                                                   label_namespace='polarity_labels',
                                                   skip_indexing=True,
                                                   num_labels=len(self.polarities))
            fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForAsteTermBiLSTM(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        words.insert(target_start_index, '#')
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [real_target_start_index + 1, real_target_end_index]
        if 'polarity' in sample:
            polarity_index = self.polarities.index(sample['polarity'])
            polarity_label_field = LabelField(polarity_index, skip_indexing=True,
                                              label_namespace='polarity_labels')
            fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForAsteTermBiLSTMWithoutSpecialToken(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        sample['target_tags'] = target_tags

        words: List = sample['words']
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index - 1

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [target_start_index, target_end_index]
        if 'polarity' in sample:
            polarity_index = self.polarities.index(sample['polarity'])
            polarity_label_field = LabelField(polarity_index, skip_indexing=True,
                                              label_namespace='polarity_labels')
            fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForTermBert(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        # __S__
        words.insert(target_start_index, '#')
        # __E__
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

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

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForAsteTermBert(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        target_tags.insert(target_start_index, 'O')
        target_tags.insert(target_end_index + 1, 'O')
        sample['target_tags'] = target_tags

        words: List = sample['words']
        # __S__
        words.insert(target_start_index, '#')
        # __E__
        if self.configuration['same_special_token']:
            words.insert(target_end_index + 1, '#')
        else:
            words.insert(target_end_index + 1, '$')
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index + 1

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

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

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            tags.insert(target_start_index, 'O')
            tags.insert(target_end_index + 1, 'O')
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [real_target_start_index + 1, real_target_end_index]
        if 'polarity' in sample:
            polarity_index = self.polarities.index(sample['polarity'])
            polarity_label_field = LabelField(polarity_index, skip_indexing=True,
                                              label_namespace='polarity_labels')
            fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForAsteTermBertWithoutSpecialToken(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        sample['target_tags'] = target_tags

        words: List = sample['words']
        sample['words'] = words

        real_target_start_index = target_start_index
        real_target_end_index = target_end_index - 1

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < real_target_start_index:
                relative_position = real_target_start_index - i
            elif i > real_target_end_index:
                relative_position = i - real_target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

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

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [target_start_index, target_end_index]
        if 'polarity' in sample:
            polarity_index = self.polarities.index(sample['polarity'])
            polarity_label_field = LabelField(polarity_index, skip_indexing=True,
                                              label_namespace='polarity_labels')
            fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForTermBertWithSecondSentence(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        words: List = sample['words']

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < target_start_index:
                relative_position = target_start_index - i
            elif i >= target_end_index:
                relative_position = i - target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word.lower())
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        bert_words.extend(self.bert_tokenizer.tokenize(' '.join(words[target_start_index: target_end_index])))
        bert_words.append('[SEP]')
        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        fields['bert'] = bert_text_field
        sample['bert_words'] = bert_words
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForAsteTermBertWithSecondSentence(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.polarities: List[str] = self.configuration['polarities'].split(',')

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        words: List = sample['words']

        sample['length'] = len(words)

        tokens = [Token(word) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)
        # relative position
        position = []
        for i in range(len(words)):
            if i < target_start_index:
                relative_position = target_start_index - i
            elif i >= target_end_index:
                relative_position = i - target_end_index
            else:
                relative_position = 0
            position.append(Token(str(relative_position)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word.lower())
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        bert_words.extend(self.bert_tokenizer.tokenize(' '.join(words[target_start_index: target_end_index])))
        bert_words.append('[SEP]')
        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        fields['bert'] = bert_text_field
        sample['bert_words'] = bert_words
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample['word_indices_of_aspect_terms'] = [target_start_index, target_end_index]
        if 'polarity' in sample:
            polarity_index = self.polarities.index(sample['polarity'])
            polarity_label_field = LabelField(polarity_index, skip_indexing=True,
                                              label_namespace='polarity_labels')
            fields["polarity_label"] = polarity_label_field

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)


class DatasetReaderForNerLstm(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, samples: List) -> Instance:
        sample = samples[0]

        fields = {}

        words: List = sample['words']
        sample['words'] = words

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        position = []
        for i in range(len(words)):
            position.append(Token(str(i)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'target_tags' in sample:
            # target_part: str = sample['metadata']['original_line'].split('####')[1]
            # tags = [e.split('=')[1] for e in target_part.split(' ')]
            # for i, tag in enumerate(tags):
            #     if not tag.startswith('T'):
            #         tags[i] = 'O'
            #     else:
            #         if i == 0 or tags[i - 1] == 'O':
            #             tags[i] = 'B'
            #         else:
            #             tags[i] = 'I'

            tags = copy.deepcopy(sample['target_tags'])
            for sample_temp in samples[1:]:
                tags_temp = sample_temp['target_tags']
                for i, tag_temp in enumerate(tags_temp):
                    if tag_temp != 'O':
                        tags[i] = tag_temp

            sample['all_target_tags'] = tags
            # for evaluation using the estimator of TOWE
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='target_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        sentence_and_samples = OrderedDict()
        for sample in samples:
            words = sample['words']
            sentence = ' '.join(words)
            if sentence not in sentence_and_samples:
                sentence_and_samples[sentence] = []
            sentence_and_samples[sentence].append(sample)
        for samples in sentence_and_samples.values():
            yield self.text_to_instance(samples)


class DatasetReaderForNerLstmOfRealASO(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        sentence: str = sample['sentence']
        instances: List = sample['instances']

        words: List = sentence.split(' ')
        sample['words'] = words

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        position = []
        for i in range(len(words)):
            position.append(Token(str(i)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'target_tags' in instances[0]:
            tags = ['O' for _ in words]
            for instance in instances:
                target_tags = instance['target_tags']
                for i, tag in enumerate(target_tags):
                    if tag != 'O':
                        tags[i] = tag

            sample['all_target_tags'] = tags
            # for evaluation using the estimator of TOWE
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='target_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        unique_sentence = OrderedDict()
        for sample in samples:
            sentence = ' '.join(sample['words'])
            if sentence not in unique_sentence:
                unique_sentence[sentence] = []
            unique_sentence[sentence].append(sample)
        for sentence, instances in unique_sentence.items():
            yield self.text_to_instance({'sentence': sentence, 'instances': instances})


class DatasetReaderForNerLstmForOTEOfRealASO(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        sentence: str = sample['sentence']
        instances: List = sample['instances']

        words: List = sentence.split(' ')
        sample['words'] = words

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        position = []
        for i in range(len(words)):
            position.append(Token(str(i)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in instances[0]:
            tags = ['O' for _ in words]
            for instance in instances:
                target_tags = instance['opinion_words_tags']
                for i, tag in enumerate(target_tags):
                    if tag != 'O':
                        tags[i] = tag[-1]

            sample['all_target_tags'] = tags
            # for evaluation using the estimator of TOWE
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='target_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        unique_sentence = OrderedDict()
        for sample in samples:
            sentence = ' '.join(sample['words'])
            if sentence not in unique_sentence:
                unique_sentence[sentence] = []
            unique_sentence[sentence].append(sample)
        for sentence, instances in unique_sentence.items():
            yield self.text_to_instance({'sentence': sentence, 'instances': instances})


class DatasetReaderForNerBertForOTEOfRealASO(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}

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

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        sentence: str = sample['sentence']
        instances: List = sample['instances']

        words: List = sentence.split(' ')
        sample['words'] = words

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        self.add_bert_words_and_word_index_bert_indices(words, fields, sample)

        position = []
        for i in range(len(words)):
            position.append(Token(str(i)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in instances[0]:
            tags = ['O' for _ in words]
            for instance in instances:
                target_tags = instance['opinion_words_tags']
                for i, tag in enumerate(target_tags):
                    if tag != 'O':
                        tags[i] = tag[-1]

            sample['all_target_tags'] = tags
            # for evaluation using the estimator of TOWE
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='target_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        unique_sentence = OrderedDict()
        for sample in samples:
            sentence = ' '.join(sample['words'])
            if sentence not in unique_sentence:
                unique_sentence[sentence] = []
            unique_sentence[sentence].append(sample)
        for sentence, instances in unique_sentence.items():
            yield self.text_to_instance({'sentence': sentence, 'instances': instances})


class DatasetReaderForNerBert(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, samples: List) -> Instance:
        sample = samples[0]
        fields = {}

        words: List = sample['words']
        sample['words'] = words

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

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

        position = []
        for i in range(len(words)):
            position.append(Token(str(i)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'target_tags' in sample:
            # target_part: str = sample['metadata']['original_line'].split('####')[1]
            # tags = [e.split('=')[1] for e in target_part.split(' ')]
            # for i, tag in enumerate(tags):
            #     if not tag.startswith('T'):
            #         tags[i] = 'O'
            #     else:
            #         if i == 0 or tags[i - 1] == 'O':
            #             tags[i] = 'B'
            #         else:
            #             tags[i] = 'I'

            tags = copy.deepcopy(sample['target_tags'])
            for sample_temp in samples[1:]:
                tags_temp = sample_temp['target_tags']
                for i, tag_temp in enumerate(tags_temp):
                    if tag_temp != 'O':
                        tags[i] = tag_temp

            sample['all_target_tags'] = tags
            # for evaluation using the estimator of TOWE
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='target_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        # unique_original_lines = set()
        # for sample in samples:
        #     if sample['metadata']['original_line'] in unique_original_lines:
        #         continue
        #     else:
        #         unique_original_lines.add(sample['metadata']['original_line'])
        #     yield self.text_to_instance(sample)
        sentence_and_samples = OrderedDict()
        for sample in samples:
            words = sample['words']
            sentence = ' '.join(words)
            if sentence not in sentence_and_samples:
                sentence_and_samples[sentence] = []
            sentence_and_samples[sentence].append(sample)
        for samples in sentence_and_samples.values():
            yield self.text_to_instance(samples)


class DatasetReaderForIOG(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    def _replace_words_with_padding_word(self, words_original: List[str], start_end_boundaries: List[List[int]],
                                         padding_word: str='@@PADDING@@'):
        """

        :param words_original:
        :param start_end_boundaries: the element in start_end_boundaries contains two elements, each is a index
        :param padding_word:
        :return:
        """
        # words = copy.deepcopy(words_original)
        words = [word.lower() for word in words_original]
        for start, end in start_end_boundaries:
            for i in range(start, end):
                words[i] = padding_word
        tokens = [Token(word) for word in words]
        return tokens

    @overrides
    def text_to_instance(self, sample: Dict) -> Instance:
        fields = {}

        target_tags: List = sample['target_tags']
        target_start_index = target_tags.index('B')
        target_end_index = target_start_index + 1
        while target_end_index < len(target_tags):
            if target_tags[target_end_index] == 'I':
                target_end_index += 1
            else:
                break

        sample['target_tags'] = target_tags

        words: List = sample['words']
        sample['words'] = words

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        left_tokens = self._replace_words_with_padding_word(words, [[target_end_index, len(words)]])
        fields['left_tokens'] = TextField(left_tokens, self.token_indexers)

        right_tokens = self._replace_words_with_padding_word(words, [[0, target_start_index]])
        fields['right_tokens'] = TextField(right_tokens, self.token_indexers)

        target_tokens = self._replace_words_with_padding_word(words,
                                                              [[0, target_start_index],
                                                               [target_end_index, len(words)]])
        fields['target_tokens'] = TextField(target_tokens, self.token_indexers)

        position = [Token(str(i)) for i in range(len(words))]
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'opinion_words_tags' in sample:
            tags: List = sample['opinion_words_tags']
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='opinion_words_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        for sample in samples:
            yield self.text_to_instance(sample)
