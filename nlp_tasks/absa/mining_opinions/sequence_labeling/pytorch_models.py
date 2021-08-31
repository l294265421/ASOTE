# -*- coding: utf-8 -*-


from typing import *
from overrides import overrides
import pickle
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField
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
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules import attention
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit
from allennlp.nn import util as allennlp_util
import dgl
from nlp_tasks.absa.mining_opinions.sequence_labeling.my_crf_tagger import CrfTagger
from nlp_tasks.absa.mining_opinions.sequence_labeling.my_simple_tagger import SimpleTagger
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import MultiHeadSelfAttention
from torch.nn import Module
import allennlp.nn.util as util
from sklearn.metrics import accuracy_score

from nlp_tasks.utils import file_utils
from nlp_tasks.utils import sequence_labeling_utils


class AttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)
        u = torch.tanh(u)
        similarities = self.uw(u)
        similarities = similarities.squeeze()
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class DotProductAttentionInHtt(nn.Module):
    """
    2016-Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.uw = nn.Linear(in_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        similarities = self.uw(h)
        similarities = similarities.squeeze()
        if self.softmax:
            alpha = allennlp_util.masked_softmax(similarities, mask)
            return alpha
        else:
            return similarities


class AverageAttention(nn.Module):
    """
    2019-emnlp-Attention is not not Explanation
    """

    def __init__(self):
        super().__init__()

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        alpha = allennlp_util.masked_softmax(mask.float(), mask)
        return alpha


class SequenceLabelingModel(Model):

    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

START_TAG = "START"
STOP_TAG = "STOP"

def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class SimpleSequenceLabelingModel(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.hidden_dim = 100
        self.embedding_dim = word_embedder.get_output_dim()
        self.dropout = 0.5

        self.tag_map = self.configuration['tag_map']
        self.tag_size = len(self.tag_map)

        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        self.transitions.data[:, self.tag_map[START_TAG]] = -1000.
        self.transitions.data[self.tag_map[STOP_TAG], :] = -1000.

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

    def real_path_score(self, logits, label):
        '''
        caculate real path score
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]

        Score = Emission_Score + Transition_Score
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])
        '''
        score = torch.zeros(1)
        label = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long), label.long()])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag_map[STOP_TAG]]
        return score

    def total_score(self, logits, label):
        """
        caculate total score

        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]

        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        obs = []
        previous = torch.full((1, self.tag_size), 0)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map[STOP_TAG]]
        # caculate total_scores
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def __viterbi_decode(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size())
        backpointers = torch.zeros(logits.size(), dtype=torch.long)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi

    def forward(self, tokens: Dict[str, torch.Tensor], labels: torch.Tensor, position: torch.Tensor,
                sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        sentences = self.word_embedder(tokens)

        batch_size = sentences.size(0)
        length = sentences.shape[1]
        embeddings = sentences.view(batch_size, length, self.embedding_dim)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = lstm_out.view(batch_size, -1, self.hidden_dim)
        logits = self.hidden2tag(lstm_out)

        lengths = [e['length'] for e in sample]
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]
            score, path = self.__viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        output = {'scores': scores, 'paths': paths}

        if labels is not None:
            real_path_score = torch.zeros(1)
            total_score = torch.zeros(1)
            for logit, tag, leng in zip(logits, labels, lengths):
                logit = logit[:leng]
                tag = tag[:leng]
                real_path_score += self.real_path_score(logit, tag)
                total_score += self.total_score(logit, tag)
            # print("total score ", total_score)
            # print("real score ", real_path_score)
            loss = total_score - real_path_score
            output['loss'] = loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
        }
        return metrics


class TCBiLSTMWithCrfTagger(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()

        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=0.2)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                          output_dim=self.hidden_size * 2,
                                          label_namespace='opinion_words_tags',
                                          dropout=None,
                                          regularizer=None)
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        max_len = embedded_text_input.shape[1]
        embedding_dim = embedded_text_input.shape[2]

        aspect_term_matrices = []
        for i, instance in enumerate(sample):
            start_index = instance['target_start_index']
            end_index = instance['target_end_index']
            word_vectors = embedded_text_input[i][start_index: end_index]
            aspect_term_vector = torch.mean(word_vectors, dim=0).unsqueeze(dim=0)
            aspect_term_matrix = aspect_term_vector.expand((max_len, embedding_dim))
            aspect_term_matrices.append(aspect_term_matrix.unsqueeze(dim=0))

        aspect_term_representation = torch.cat(aspect_term_matrices, dim=0)
        aspect_term_representation = self.dropout(aspect_term_representation)

        lstm_input = torch.cat([embedded_text_input, aspect_term_representation], dim=-1)

        lstm_result, _ = self.lstm(lstm_input)

        encoded_text = lstm_result

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class TermBiLSTM(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class AsteTermBiLSTM(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            self.sentiment_specific_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                                   num_layers=self.configuration['lstm_layer_num_of_sentiment_specific'],
                                                   bidirectional=True, batch_first=True,
                                                   dropout=0.5)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

        self.polarity_num = len(self.configuration['polarities'].split(','))
        self.sentiment_fc = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size * 2, self.polarity_num))
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, polarity_label: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        towe_result = self._tagger_ner.forward(**input_for_crf_tagger)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            lstm_result, _ = self.sentiment_specific_lstm(lstm_result)
            lstm_result = self.dropout(lstm_result)

        sentiment_outputs = []
        for i, element in enumerate(sample):
            word_indices_of_aspect_term = element['word_indices_of_aspect_terms']
            start_index = word_indices_of_aspect_term[0]
            end_index = word_indices_of_aspect_term[1]
            word_representations = lstm_result[i][start_index: end_index]
            aspect_term_word_num = end_index - start_index
            if aspect_term_word_num > 1:
                aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                aspect_term_representation = aspect_term_representation.unsqueeze(0)
            else:
                aspect_term_representation = word_representations
            sentiment_output = self.sentiment_fc(aspect_term_representation)
            sentiment_outputs.append(sentiment_output)
        sentiment_outputs_cat = torch.cat(sentiment_outputs, dim=0)
        atsa_result = {}
        if polarity_label is not None:
            loss = self.sentiment_loss(sentiment_outputs_cat, polarity_label.long())
            if torch.isnan(loss):
                print()

            self._accuracy(sentiment_outputs_cat, polarity_label)

            atsa_result['logit'] = sentiment_outputs_cat
            atsa_result['label'] = polarity_label
            atsa_result['loss'] = loss

        result = {
            'towe_result': towe_result,
            'atsa_result': atsa_result,
        }
        joint_mode = self.configuration['joint_mode']
        if joint_mode == 'towe':
            result['loss'] = towe_result['loss']
        elif joint_mode == 'atsa':
            result = atsa_result['loss']
        else:
            result['loss'] = towe_result['loss'] + atsa_result['loss']

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class AsoTermBiLSTM(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, polarity_label: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        so_result = self._tagger_ner.forward(**input_for_crf_tagger)
        return so_result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class AsoTermBiLSTMBert(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.bert_word_embedder = bert_word_embedder

        self.embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def get_bert_embedding(self, bert, sample, word_embeddings_size):
        bert_mask = bert['mask']
        token_type_ids = bert['bert-type-ids']
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
        return aspect_word_embeddings_from_bert_cat

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, polarity_label: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        word_embeddings_size = embedded_text_input.size()
        embedded_text_input = self.get_bert_embedding(bert, sample, word_embeddings_size)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        so_result = self._tagger_ner.forward(**input_for_crf_tagger)
        return so_result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class AsoTermBiLSTMBertWithPosition(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.bert_word_embedder = bert_word_embedder

        self.embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def get_bert_embedding(self, bert, sample, word_embeddings_size, bert_position: torch.Tensor):
        bert_mask = bert['mask']
        token_type_ids = bert['bert-type-ids']
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets,
                                                       position_ids=bert_position.long())

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
        return aspect_word_embeddings_from_bert_cat

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, bert_position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, polarity_label: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        word_embeddings_size = embedded_text_input.size()
        embedded_text_input = self.get_bert_embedding(bert, sample, word_embeddings_size, bert_position)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        so_result = self._tagger_ner.forward(**input_for_crf_tagger)
        return so_result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class WarmupSequenceLabelingModel(SequenceLabelingModel):

    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)
        self.loss_weights = {
            'atsa_loss_weight': 1,
            'towe_loss_weight': 1
        }

    def matrix_mul(self, input, weight, bias=False):
        feature_list = []
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)

        return torch.cat(feature_list, 0).squeeze()

    def element_wise_mul(self, input1, input2, return_not_sum_result=False):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            feature_2 = feature_2.unsqueeze(1)
            feature_2 = feature_2.expand_as(feature_1)
            feature = feature_1 * feature_2
            feature = feature.unsqueeze(0)
            feature_list.append(feature)
        output = torch.cat(feature_list, 0)

        result = torch.sum(output, 1)
        if return_not_sum_result:
            return result, output
        else:
            return result

    def towe_layers(self):
        pass


class AsteTermBiLSTMWithSLA(WarmupSequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            self.sentiment_specific_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                                   num_layers=self.configuration['lstm_layer_num_of_sentiment_specific'],
                                                   bidirectional=True, batch_first=True,
                                                   dropout=0.5)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

        self.polarity_num = len(self.configuration['polarities'].split(','))
        if self.configuration['merge_mode'] == 'concat':
            self.sentiment_fc = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_size * 2, self.polarity_num))
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_size * 2, self.polarity_num))
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        if self.configuration['use_different_encoder']:
            self.sentiment_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                          num_layers=1,
                                          bidirectional=True, batch_first=True
                                          )

    def towe_layers(self):
        result = []
        result.append(self.word_embedder)
        result.append(self.position_embedder)
        result.append(self.lstm)
        result.append(self.feedforward)
        result.append(self._tagger_ner)
        return result

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, polarity_label: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        towe_result = self._tagger_ner.forward(**input_for_crf_tagger)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            lstm_result, _ = self.sentiment_specific_lstm(lstm_result)
            lstm_result = self.dropout(lstm_result)

        if self.configuration['use_different_encoder']:
            lstm_result, _ = self.sentiment_lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)

        # sequence label attention
        opinion_tags = self._tagger_ner.vocab._token_to_index['opinion_words_tags']
        other_index = opinion_tags['O']

        if self.configuration['grad_communication']:
            towe_logits = towe_result['logits']
        else:
            towe_logits = towe_result['logits'].detach()
        if self.configuration['softmax_after_opinion_logit']:
            towe_prob = torch.softmax(towe_logits, dim=-1)
        else:
            towe_prob = towe_logits
        towe_prob_mask = torch.ones_like(towe_prob)
        towe_prob_mask[:, :, other_index] = 0
        towe_prob_attention = towe_prob * towe_prob_mask
        towe_prob_attention = torch.sum(towe_prob_attention, dim=-1)
        towe_prob_attention = allennlp_util.masked_softmax(towe_prob_attention, mask)
        sentiment_representations_from_towe = self.element_wise_mul(lstm_result, towe_prob_attention)

        if self.configuration['output_attention']:
            towe_prob_attention_shape = towe_prob_attention.shape
            for i in range(towe_prob_attention_shape[0]):
                temp = towe_prob_attention[i].detach().cpu().numpy().tolist()
                words = sample[i]['words']
                opinion_words_tags = sample[i]['opinion_words_tags']
                target_tags = sample[i]['target_tags']
                print(' '.join(words))
                print(['%d-%s-%.3f-O:%s-T:%s' % (j, words[j], temp[j], opinion_words_tags[j], target_tags[j]) for j in range(len(words))])

        sentiment_representations = []
        for i, element in enumerate(sample):
            word_indices_of_aspect_term = element['word_indices_of_aspect_terms']
            start_index = word_indices_of_aspect_term[0]
            end_index = word_indices_of_aspect_term[1]
            word_representations = lstm_result[i][start_index: end_index]
            aspect_term_word_num = end_index - start_index
            if aspect_term_word_num > 1:
                aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                aspect_term_representation = aspect_term_representation.unsqueeze(0)
            else:
                aspect_term_representation = word_representations
            sentiment_representations.append(aspect_term_representation)
        sentiment_representations_cat = torch.cat(sentiment_representations, dim=0)

        if self.configuration['merge_mode'] == 'sum':
            sentiment_representations_merge = sentiment_representations_from_towe + sentiment_representations_cat
        elif self.configuration['merge_mode'] == 'mean':
            sentiment_representations_merge = (sentiment_representations_from_towe + sentiment_representations_cat) / 2
        elif self.configuration['merge_mode'] == 'concat':
            sentiment_representations_merge = torch.cat([sentiment_representations_from_towe, sentiment_representations_cat], dim=-1)
        else:
            raise NotImplementedError(self.configuration['merge_mode'])

        sentiment_outputs_cat = self.sentiment_fc(sentiment_representations_merge)
        atsa_result = {}
        if polarity_label is not None:
            loss = self.sentiment_loss(sentiment_outputs_cat, polarity_label.long())
            if torch.isnan(loss):
                print()

            self._accuracy(sentiment_outputs_cat, polarity_label)

            atsa_result['logit'] = sentiment_outputs_cat
            atsa_result['label'] = polarity_label
            atsa_result['loss'] = loss

        result = {
            'towe_result': towe_result,
            'atsa_result': atsa_result,
        }
        joint_mode = self.configuration['joint_mode']
        if joint_mode == 'towe':
            result['loss'] = towe_result['loss']
        elif joint_mode == 'atsa':
            result = atsa_result['loss']
        else:
            result['loss'] = 0
            if self.loss_weights['towe_loss_weight'] != 0:
                result['loss'] += self.loss_weights['towe_loss_weight'] * towe_result['loss']
            if self.loss_weights['atsa_loss_weight'] != 0:
                result['loss'] += self.loss_weights['atsa_loss_weight'] * atsa_result['loss']

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result
    
    
class MILForASO(WarmupSequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            self.sentiment_specific_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                                   num_layers=self.configuration['lstm_layer_num_of_sentiment_specific'],
                                                   bidirectional=True, batch_first=True,
                                                   dropout=0.5)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

        self.polarity_num = len(self.configuration['polarities'].split(','))
        self.sentiment_fc = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size * 2, self.polarity_num))
        # self.sentiment_loss = nn.CrossEntropyLoss()
        self.sentiment_loss = nn.BCEWithLogitsLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        if self.configuration['use_different_encoder']:
            self.sentiment_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                          num_layers=1,
                                          bidirectional=True, batch_first=True
                                          )

    def towe_layers(self):
        result = []
        result.append(self.word_embedder)
        result.append(self.position_embedder)
        result.append(self.lstm)
        result.append(self.feedforward)
        result.append(self._tagger_ner)
        return result

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, polarity_label: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        towe_result = self._tagger_ner.forward(**input_for_crf_tagger)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            lstm_result, _ = self.sentiment_specific_lstm(lstm_result)
            lstm_result = self.dropout(lstm_result)

        if self.configuration['use_different_encoder']:
            lstm_result, _ = self.sentiment_lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        sentiment_outputs_of_words = self.sentiment_fc(lstm_result)

        # sequence label attention
        opinion_tags = self._tagger_ner.vocab._token_to_index['opinion_words_tags']
        other_index = opinion_tags['O']

        if self.configuration['grad_communication']:
            towe_logits = towe_result['logits']
        else:
            towe_logits = towe_result['logits'].detach()
        if self.configuration['softmax_after_opinion_logit']:
            towe_prob = torch.softmax(towe_logits, dim=-1)
        else:
            towe_prob = towe_logits
        towe_prob_mask = torch.ones_like(towe_prob)
        towe_prob_mask[:, :, other_index] = 0
        towe_prob_attention = towe_prob * towe_prob_mask
        towe_prob_attention = torch.sum(towe_prob_attention, dim=-1)
        towe_prob_attention = allennlp_util.masked_softmax(towe_prob_attention, mask)

        sentiment_outputs_from_towe = self.element_wise_mul(sentiment_outputs_of_words, towe_prob_attention)

        if self.configuration['output_attention']:
            towe_prob_attention_shape = towe_prob_attention.shape
            for i in range(towe_prob_attention_shape[0]):
                temp = towe_prob_attention[i].detach().cpu().numpy().tolist()
                words = sample[i]['words']
                opinion_words_tags = sample[i]['opinion_words_tags']
                target_tags = sample[i]['target_tags']
                print(' '.join(words))
                print(['%d-%s-%.3f-O:%s-T:%s' % (j, words[j], temp[j], opinion_words_tags[j], target_tags[j]) for j in range(len(words))])

        sentiment_representations = []
        for i, element in enumerate(sample):
            word_indices_of_aspect_term = element['word_indices_of_aspect_terms']
            start_index = word_indices_of_aspect_term[0]
            end_index = word_indices_of_aspect_term[1]
            word_representations = lstm_result[i][start_index: end_index]
            aspect_term_word_num = end_index - start_index
            if aspect_term_word_num > 1:
                aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                aspect_term_representation = aspect_term_representation.unsqueeze(0)
            else:
                aspect_term_representation = word_representations
            sentiment_representations.append(aspect_term_representation)
        sentiment_representations_cat = torch.cat(sentiment_representations, dim=0)

        sentiment_outputs_cat = self.sentiment_fc(sentiment_representations_cat)

        sentiment_outputs = (sentiment_outputs_from_towe + sentiment_outputs_cat) / 2

        atsa_result = {
            'sentiment_outputs_of_words': sentiment_outputs_of_words,
            'towe_attention': towe_prob_attention
        }
        if polarity_label is not None:
            loss = self.sentiment_loss(sentiment_outputs, polarity_label.float())
            if torch.isnan(loss):
                print()

            # self._accuracy(sentiment_outputs_cat, polarity_label)

            atsa_result['logit'] = sentiment_outputs_cat
            atsa_result['label'] = polarity_label
            atsa_result['loss'] = loss

        result = {
            'towe_result': towe_result,
            'atsa_result': atsa_result
        }
        joint_mode = self.configuration['joint_mode']
        if joint_mode == 'towe':
            result['loss'] = towe_result['loss']
        elif joint_mode == 'atsa':
            result = atsa_result['loss']
        else:
            result['loss'] = 0
            if self.loss_weights['towe_loss_weight'] != 0:
                result['loss'] += self.loss_weights['towe_loss_weight'] * towe_result['loss']
            if self.loss_weights['atsa_loss_weight'] != 0:
                result['loss'] += self.loss_weights['atsa_loss_weight'] * atsa_result['loss']

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class MILForASOBert(WarmupSequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict,
                 bert_word_embedder: TextFieldEmbedder=None,
                 another_bert_word_embedder: TextFieldEmbedder = None
                 ):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.bert_word_embedder = bert_word_embedder
        self.another_bert_word_embedder = another_bert_word_embedder

        self.embedding_dim = bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            self.sentiment_specific_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                                   num_layers=self.configuration['lstm_layer_num_of_sentiment_specific'],
                                                   bidirectional=True, batch_first=True,
                                                   dropout=0.5)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

        self.polarity_num = len(self.configuration['polarities'].split(','))
        self.sentiment_fc = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size * 2, self.polarity_num))
        # self.sentiment_loss = nn.CrossEntropyLoss()
        self.sentiment_loss = nn.BCEWithLogitsLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        if self.configuration['use_different_encoder']:
            self.sentiment_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                          num_layers=1,
                                          bidirectional=True, batch_first=True
                                          )

    def towe_layers(self):
        result = []
        result.append(self.word_embedder)
        result.append(self.position_embedder)
        result.append(self.lstm)
        result.append(self.feedforward)
        result.append(self._tagger_ner)
        return result

    def get_bert_embedding(self, bert, sample, word_embeddings_size, bert_word_embedder):
        bert_mask = bert['mask']
        token_type_ids = bert['bert-type-ids']
        offsets = bert['bert-offsets']
        bert_word_embeddings = bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
        return aspect_word_embeddings_from_bert_cat

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, polarity_label: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        word_embeddings_size = embedded_text_input.size()
        embedded_text_input = self.get_bert_embedding(bert, sample, word_embeddings_size, self.bert_word_embedder)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        towe_result = self._tagger_ner.forward(**input_for_crf_tagger)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            lstm_result, _ = self.sentiment_specific_lstm(lstm_result)
            lstm_result = self.dropout(lstm_result)

        if self.configuration['use_different_encoder']:
            embedded_text_input = self.get_bert_embedding(bert, sample, word_embeddings_size,
                                                          self.another_bert_word_embedder)

            if self.configuration['position']:
                position_input = self.position_embedder(position)
                lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
            else:
                lstm_input = embedded_text_input

            lstm_input = self.dropout(lstm_input)
            lstm_result, _ = self.sentiment_lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        sentiment_outputs_of_words = self.sentiment_fc(lstm_result)

        # sequence label attention
        opinion_tags = self._tagger_ner.vocab._token_to_index['opinion_words_tags']
        other_index = opinion_tags['O']

        if self.configuration['grad_communication']:
            towe_logits = towe_result['logits']
        else:
            towe_logits = towe_result['logits'].detach()
        if self.configuration['softmax_after_opinion_logit']:
            towe_prob = torch.softmax(towe_logits, dim=-1)
        else:
            towe_prob = towe_logits
        towe_prob_mask = torch.ones_like(towe_prob)
        towe_prob_mask[:, :, other_index] = 0
        towe_prob_attention = towe_prob * towe_prob_mask
        towe_prob_attention = torch.sum(towe_prob_attention, dim=-1)
        towe_prob_attention = allennlp_util.masked_softmax(towe_prob_attention, mask)

        sentiment_outputs_from_towe = self.element_wise_mul(sentiment_outputs_of_words, towe_prob_attention)

        if self.configuration['output_attention']:
            towe_prob_attention_shape = towe_prob_attention.shape
            for i in range(towe_prob_attention_shape[0]):
                temp = towe_prob_attention[i].detach().cpu().numpy().tolist()
                words = sample[i]['words']
                opinion_words_tags = sample[i]['opinion_words_tags']
                target_tags = sample[i]['target_tags']
                print(' '.join(words))
                print(['%d-%s-%.3f-O:%s-T:%s' % (j, words[j], temp[j], opinion_words_tags[j], target_tags[j]) for j in range(len(words))])

        sentiment_representations = []
        for i, element in enumerate(sample):
            word_indices_of_aspect_term = element['word_indices_of_aspect_terms']
            start_index = word_indices_of_aspect_term[0]
            end_index = word_indices_of_aspect_term[1]
            word_representations = lstm_result[i][start_index: end_index]
            aspect_term_word_num = end_index - start_index
            if aspect_term_word_num > 1:
                aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                aspect_term_representation = aspect_term_representation.unsqueeze(0)
            else:
                aspect_term_representation = word_representations
            sentiment_representations.append(aspect_term_representation)
        sentiment_representations_cat = torch.cat(sentiment_representations, dim=0)

        sentiment_outputs_cat = self.sentiment_fc(sentiment_representations_cat)

        sentiment_outputs = (sentiment_outputs_from_towe + sentiment_outputs_cat) / 2

        atsa_result = {
            'sentiment_outputs_of_words': sentiment_outputs_of_words,
            'towe_attention': towe_prob_attention
        }
        if polarity_label is not None:
            loss = self.sentiment_loss(sentiment_outputs, polarity_label.float())
            if torch.isnan(loss):
                print()

            # self._accuracy(sentiment_outputs_cat, polarity_label)

            atsa_result['logit'] = sentiment_outputs_cat
            atsa_result['label'] = polarity_label
            atsa_result['loss'] = loss

        result = {
            'towe_result': towe_result,
            'atsa_result': atsa_result
        }
        joint_mode = self.configuration['joint_mode']
        if joint_mode == 'towe':
            result['loss'] = towe_result['loss']
        elif joint_mode == 'atsa':
            result = atsa_result['loss']
        else:
            result['loss'] = 0
            if self.loss_weights['towe_loss_weight'] != 0:
                result['loss'] += self.loss_weights['towe_loss_weight'] * towe_result['loss']
            if self.loss_weights['atsa_loss_weight'] != 0:
                result['loss'] += self.loss_weights['atsa_loss_weight'] * atsa_result['loss']

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class TermBert(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.bert_word_embedder = bert_word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                num_layers=self.configuration['lstm_layer_num_in_bert'],
                                bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        word_embeddings_size = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)

        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([aspect_word_embeddings_from_bert_cat, position_input], dim=-1)
        else:
            lstm_input = aspect_word_embeddings_from_bert_cat

        lstm_input = self.dropout(lstm_input)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        else:
            lstm_result = lstm_input

        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class TermBertWithPosition(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.bert_word_embedder = bert_word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                num_layers=self.configuration['lstm_layer_num_in_bert'],
                                bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, bert_position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        word_embeddings_size = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)

        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets,
                                                       position_ids=bert_position.long())

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([aspect_word_embeddings_from_bert_cat, position_input], dim=-1)
        else:
            lstm_input = aspect_word_embeddings_from_bert_cat

        lstm_input = self.dropout(lstm_input)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        else:
            lstm_result = lstm_input

        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class TermBiLSTMWithSecondSentence(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.bert_word_embedder = bert_word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        # self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()
        self.bert_embedding_dim = self.embedding_dim
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                num_layers=self.configuration['lstm_layer_num_in_bert'],
                                bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, bert_position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        word_embeddings_size = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)

        # bert_mask = bert['mask']
        # # bert_word_embeddings = self.bert_word_embedder(bert)
        # token_type_ids = bert['bert-type-ids']
        # # token_type_ids_size = token_type_ids.size()
        # # for i in range(token_type_ids_size[1]):
        # #     print(token_type_ids[0][i])
        # offsets = bert['bert-offsets']
        # bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets,
        #                                                position_ids=bert_position.long())

        bert_word_embeddings = self.word_embedder(bert)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([aspect_word_embeddings_from_bert_cat, position_input], dim=-1)
        else:
            lstm_input = aspect_word_embeddings_from_bert_cat

        lstm_input = self.dropout(lstm_input)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        else:
            lstm_result = lstm_input

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class AsteTermBert(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.bert_word_embedder = bert_word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                num_layers=self.configuration['lstm_layer_num_in_bert'],
                                bidirectional=True, batch_first=True)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            self.sentiment_specific_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                                   num_layers=self.configuration[
                                                       'lstm_layer_num_of_sentiment_specific'],
                                                   bidirectional=True, batch_first=True,
                                                   dropout=0.5)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None, polarity_label: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        word_embeddings_size = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)

        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([aspect_word_embeddings_from_bert_cat, position_input], dim=-1)
        else:
            lstm_input = aspect_word_embeddings_from_bert_cat

        lstm_input = self.dropout(lstm_input)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        else:
            lstm_result = lstm_input

        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        towe_result = self._tagger_ner.forward(**input_for_crf_tagger)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            lstm_result, _ = self.sentiment_specific_lstm(lstm_result)
            lstm_result = self.dropout(lstm_result)

        sentiment_outputs = []
        for i, element in enumerate(sample):
            word_indices_of_aspect_term = element['word_indices_of_aspect_terms']
            start_index = word_indices_of_aspect_term[0]
            end_index = word_indices_of_aspect_term[1]
            word_representations = lstm_result[i][start_index: end_index]
            aspect_term_word_num = end_index - start_index
            if aspect_term_word_num > 1:
                aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                aspect_term_representation = aspect_term_representation.unsqueeze(0)
            else:
                aspect_term_representation = word_representations
            sentiment_output = self.sentiment_fc(aspect_term_representation)
            sentiment_outputs.append(sentiment_output)
        sentiment_outputs_cat = torch.cat(sentiment_outputs, dim=0)
        atsa_result = {}
        if polarity_label is not None:
            loss = self.sentiment_loss(sentiment_outputs_cat, polarity_label.long())
            if torch.isnan(loss):
                print()

            self._accuracy(sentiment_outputs_cat, polarity_label)

            atsa_result['logit'] = sentiment_outputs_cat
            atsa_result['label'] = polarity_label
            atsa_result['loss'] = loss

        result = {
            'towe_result': towe_result,
            'atsa_result': atsa_result,
        }
        joint_mode = self.configuration['joint_mode']
        if joint_mode == 'towe':
            result['loss'] = towe_result['loss']
        elif joint_mode == 'atsa':
            result = atsa_result['loss']
        else:
            result['loss'] = towe_result['loss'] + atsa_result['loss']

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class AsteTermBertWithSLA(WarmupSequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None,
                 another_bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.bert_word_embedder = bert_word_embedder
        self.another_bert_word_embedder = another_bert_word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                num_layers=self.configuration['lstm_layer_num_in_bert'],
                                bidirectional=True, batch_first=True)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            self.sentiment_specific_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size,
                                                   num_layers=self.configuration[
                                                       'lstm_layer_num_of_sentiment_specific'],
                                                   bidirectional=True, batch_first=True,
                                                   dropout=0.5)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='opinion_words_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

        self.polarity_num = len(self.configuration['polarities'].split(','))
        if self.configuration['merge_mode'] == 'concat':
            self.sentiment_fc = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_size * 2, self.polarity_num))
        else:
            self.sentiment_fc = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_size * 2, self.polarity_num))
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        if self.configuration['use_different_encoder']:
            self.sentiment_lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                          num_layers=1,
                                          bidirectional=True, batch_first=True
                                          )

    def towe_layers(self):
        result = []
        result.append(self.word_embedder)
        result.append(self.bert_word_embedder)
        result.append(self.position_embedder)
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            result.append(self.lstm)
        result.append(self.feedforward)
        result.append(self._tagger_ner)
        return result

    def generate_bert_embedding(self, bert_word_embedder, bert: torch.Tensor,
                                sample: list, word_embeddings_size):
        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
        return aspect_word_embeddings_from_bert_cat

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None, polarity_label: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        word_embeddings_size = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)

        aspect_word_embeddings_from_bert_cat = self.generate_bert_embedding(self.bert_word_embedder, bert, sample,
                                                                            word_embeddings_size)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([aspect_word_embeddings_from_bert_cat, position_input], dim=-1)
        else:
            lstm_input = aspect_word_embeddings_from_bert_cat

        lstm_input = self.dropout(lstm_input)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        else:
            lstm_result = lstm_input

        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        towe_result = self._tagger_ner.forward(**input_for_crf_tagger)

        if self.configuration['lstm_layer_num_of_sentiment_specific'] != 0:
            lstm_result, _ = self.sentiment_specific_lstm(lstm_result)
            lstm_result = self.dropout(lstm_result)

        if self.configuration['use_different_encoder']:
            lstm_input = self.generate_bert_embedding(self.another_bert_word_embedder, bert, sample,
                                                      word_embeddings_size)
            lstm_input = self.dropout(lstm_input)

            if self.configuration['lstm_layer_num_in_bert'] != 0:
                lstm_result, _ = self.sentiment_lstm(lstm_input)
                lstm_result = self.dropout(lstm_result)
            else:
                lstm_result = lstm_input

        # sequence label attention
        opinion_tags = self._tagger_ner.vocab._token_to_index['opinion_words_tags']
        other_index = opinion_tags['O']

        if self.configuration['grad_communication']:
            towe_logits = towe_result['logits']
        else:
            towe_logits = towe_result['logits'].detach()
        if self.configuration['softmax_after_opinion_logit']:
            towe_prob = torch.softmax(towe_logits, dim=-1)
        else:
            towe_prob = towe_logits
        towe_prob_mask = torch.ones_like(towe_prob)
        towe_prob_mask[:, :, other_index] = 0
        towe_prob_attention = towe_prob * towe_prob_mask
        towe_prob_attention = torch.sum(towe_prob_attention, dim=-1)
        towe_prob_attention = allennlp_util.masked_softmax(towe_prob_attention, mask)
        sentiment_representations_from_towe = self.element_wise_mul(lstm_result, towe_prob_attention)

        if self.configuration['output_attention']:
            towe_prob_attention_shape = towe_prob_attention.shape
            for i in range(towe_prob_attention_shape[0]):
                temp = towe_prob_attention[i].detach().cpu().numpy().tolist()
                words = sample[i]['words']
                opinion_words_tags = sample[i]['opinion_words_tags']
                target_tags = sample[i]['target_tags']
                print(' '.join(words))
                print(['%d-%s-%.3f-O:%s-T:%s' % (j, words[j], temp[j], opinion_words_tags[j], target_tags[j]) for j in
                       range(len(words))])

        sentiment_representations = []
        for i, element in enumerate(sample):
            word_indices_of_aspect_term = element['word_indices_of_aspect_terms']
            start_index = word_indices_of_aspect_term[0]
            end_index = word_indices_of_aspect_term[1]
            word_representations = lstm_result[i][start_index: end_index]
            aspect_term_word_num = end_index - start_index
            if aspect_term_word_num > 1:
                aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                aspect_term_representation = aspect_term_representation.unsqueeze(0)
            else:
                aspect_term_representation = word_representations
            sentiment_representations.append(aspect_term_representation)
        sentiment_representations_cat = torch.cat(sentiment_representations, dim=0)

        if self.configuration['merge_mode'] == 'sum':
            sentiment_representations_merge = sentiment_representations_from_towe + sentiment_representations_cat
        elif self.configuration['merge_mode'] == 'mean':
            sentiment_representations_merge = (sentiment_representations_from_towe + sentiment_representations_cat) / 2
        elif self.configuration['merge_mode'] == 'concat':
            sentiment_representations_merge = torch.cat(
                [sentiment_representations_from_towe, sentiment_representations_cat], dim=-1)
        else:
            raise NotImplementedError(self.configuration['merge_mode'])

        sentiment_outputs_cat = self.sentiment_fc(sentiment_representations_merge)

        atsa_result = {}
        if polarity_label is not None:
            loss = self.sentiment_loss(sentiment_outputs_cat, polarity_label.long())
            if torch.isnan(loss):
                print()

            self._accuracy(sentiment_outputs_cat, polarity_label)

            atsa_result['logit'] = sentiment_outputs_cat
            atsa_result['label'] = polarity_label
            atsa_result['loss'] = loss

        result = {
            'towe_result': towe_result,
            'atsa_result': atsa_result,
        }
        joint_mode = self.configuration['joint_mode']
        if joint_mode == 'towe':
            result['loss'] = towe_result['loss']
        elif joint_mode == 'atsa':
            result = atsa_result['loss']
        else:
            result['loss'] = 0
            if self.loss_weights['towe_loss_weight'] != 0:
                result['loss'] += self.loss_weights['towe_loss_weight'] * towe_result['loss']
            if self.loss_weights['atsa_loss_weight'] != 0:
                result['loss'] += self.loss_weights['atsa_loss_weight'] * atsa_result['loss']

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class NerLstm(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.embedding_dim
        self.hidden_size = self.embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='target_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='target_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class NerBertForOTE(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.bert_word_embedder = bert_word_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()
        self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='target_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='target_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def get_bert_embedding(self, bert, sample, word_embeddings_size):
        bert_mask = bert['mask']
        token_type_ids = bert['bert-type-ids']
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
        return aspect_word_embeddings_from_bert_cat

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        word_embeddings_size = embedded_text_input.size()
        embedded_text_input = self.get_bert_embedding(bert, sample, word_embeddings_size)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([embedded_text_input, position_input], dim=-1)
        else:
            lstm_input = embedded_text_input

        lstm_input = self.dropout(lstm_input)

        lstm_result, _ = self.lstm(lstm_input)
        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class NerBert(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.bert_word_embedder = bert_word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                num_layers=self.configuration['lstm_layer_num_in_bert'],
                                bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='target_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='target_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        word_embeddings_size = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)

        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([aspect_word_embeddings_from_bert_cat, position_input], dim=-1)
        else:
            lstm_input = aspect_word_embeddings_from_bert_cat

        lstm_input = self.dropout(lstm_input)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        else:
            lstm_result = lstm_input

        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class IOG(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()

        self.input_size = self.embedding_dim
        self.hidden_size = 200
        self.rnn_L = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn_R = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn_global = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=True,
                                  batch_first=True)

        if self.configuration['crf']:
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 4,
                                              label_namespace='opinion_words_tags',
                                              dropout=None,
                                              regularizer=None)
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 4,
                                                    label_namespace='opinion_words_tags',
                                                    regularizer=None
                                                    )

        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward_backup(self, tokens: Dict[str, torch.Tensor], left_tokens: Dict[str, torch.Tensor],
                right_tokens: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.Tensor],
                position: torch.Tensor, sample: list, labels: torch.Tensor=None) -> torch.Tensor:
        sentence = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        global_encoded, _ = self.rnn_global(sentence)

        left_sentence = self.word_embedder(left_tokens)
        left_encoded, temp1 = self.rnn_L(left_sentence)
        left_shape = list(left_encoded.shape)
        left_encoded_separated = left_encoded.view([left_shape[0], left_shape[1], 2, left_shape[2] // 2])
        left_forward = left_encoded_separated[:, :, 0, :]
        left_backward = left_encoded_separated[:, :, 1, :]

        right_sentence = self.word_embedder(right_tokens)
        right_encoded, _ = self.rnn_R(right_sentence)
        right_shape = list(right_encoded.shape)
        right_encoded_separated = right_encoded.view([right_shape[0], right_shape[1], 2, right_shape[2] // 2])
        right_forward = right_encoded_separated[:, :, 0, :]
        right_backward = right_encoded_separated[:, :, 1, :]

        inward = left_forward + right_backward

        outward = left_backward + right_forward
        ioward = torch.cat([inward, outward], dim=-1)

        target_mask = util.get_text_field_mask(target_tokens)
        scale = 1 - 0.5 * target_mask
        scale_unsqueezed = scale.unsqueeze(dim=-1)
        scale_expanded = scale_unsqueezed.expand_as(ioward)

        final_ioward = ioward * scale_expanded

        encoded_text = torch.cat([global_encoded, final_ioward], dim=-1)
        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def forward(self, tokens: Dict[str, torch.Tensor], left_tokens: Dict[str, torch.Tensor],
                right_tokens: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.Tensor],
                position: torch.Tensor, sample: list, labels: torch.Tensor=None) -> torch.Tensor:
        # sentence = self.word_embedder(tokens)
        # mask = util.get_text_field_mask(tokens)
        # global_encoded, _ = self.rnn_global(sentence)
        #
        # left_sentence = self.word_embedder(left_tokens)
        # left_encoded, temp1 = self.rnn_L(left_sentence)
        # left_shape = list(left_encoded.shape)
        # left_encoded_separated = left_encoded.view([left_shape[0], left_shape[1], 2, left_shape[2] // 2])
        # left_forward = left_encoded_separated[:, :, 0, :]
        # left_backward = left_encoded_separated[:, :, 1, :]
        #
        # right_sentence = self.word_embedder(right_tokens)
        # right_encoded, _ = self.rnn_R(right_sentence)
        # right_shape = list(right_encoded.shape)
        # right_encoded_separated = right_encoded.view([right_shape[0], right_shape[1], 2, right_shape[2] // 2])
        # right_forward = right_encoded_separated[:, :, 0, :]
        # right_backward = right_encoded_separated[:, :, 1, :]
        #
        # inward = left_forward + right_backward
        #
        # outward = left_backward + right_forward
        # ioward = torch.cat([inward, outward], dim=-1)
        #
        # target_mask = util.get_text_field_mask(target_tokens)
        # scale = 1 - 0.5 * target_mask
        # scale_unsqueezed = scale.unsqueeze(dim=-1)
        # scale_expanded = scale_unsqueezed.expand_as(ioward)
        #
        # final_ioward = ioward * scale_expanded
        #
        # encoded_text = torch.cat([global_encoded, final_ioward], dim=-1)


        # target = batch.target
        # sentence = self.word_rep(batch)
        sentence = self.word_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        # left_mask = batch.left_mask
        left_mask = util.get_text_field_mask(left_tokens)
        # right_mask = batch.right_mask
        right_mask = util.get_text_field_mask(right_tokens)
        # target_mask = target != 0
        target_mask = util.get_text_field_mask(target_tokens)

        left_context = sentence * left_mask.unsqueeze(-1).float().expand_as(sentence)
        right_context = sentence * right_mask.unsqueeze(-1).float().expand_as(sentence)

        left_encoded, _ = self.rnn_L(left_context)
        right_encoded, _ = self.rnn_R(right_context)
        global_encoded, _ = self.rnn_global(sentence)

        left_encoded = left_encoded * left_mask.unsqueeze(-1).float().expand_as(left_encoded)
        right_encoded = right_encoded * right_mask.unsqueeze(-1).float().expand_as(right_encoded)

        encoded = left_encoded + right_encoded
        target_average_mask = 1 - 1 / 2 * target_mask.unsqueeze(-1).float().expand_as(encoded)
        encoded = encoded * target_average_mask

        encoded_text = torch.cat((encoded, global_encoded), dim=-1)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


class Estimator:

    def estimate(self, ds: Iterable[Instance]) -> dict:
        raise NotImplementedError('estimate')


class SequenceLabelingModelEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration

    def score_BIO(self, predicted, golden, ignore_index=-1):
        # tag2id = {'B': 1, 'I': 2, 'O': 0}
        assert len(predicted) == len(golden)
        sum_all = 0
        sum_correct = 0
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        # print(predicted)
        # print(golden)
        for i in range(len(golden)):
            length = len(golden[i])
            # print(length)
            # print(predicted[i])
            # print(golden[i])
            golden_01 = 0
            correct_01 = 0
            predict_01 = 0
            predict_items = []
            golden_items = []
            golden_seq = []
            predict_seq = []
            golden_i = golden[i]
            predicted_i = predicted[i]
            for j in range(length):
                if golden[i][j] == ignore_index:
                    break
                if golden[i][j] == 'B':
                    if len(golden_seq) > 0:  # 00
                        golden_items.append(golden_seq)
                        golden_seq = []
                    golden_seq.append(j)
                elif golden[i][j] == 'I':
                    if len(golden_seq) > 0:
                        golden_seq.append(j)
                elif golden[i][j] == 'O':
                    if len(golden_seq) > 0:
                        golden_items.append(golden_seq)
                        golden_seq = []
                if predicted[i][j] == 'B':
                    if len(predict_seq) > 0:  # 00
                        predict_items.append(predict_seq)
                        predict_seq = []
                    predict_seq.append(j)
                elif predicted[i][j] == 'I':
                    if len(predict_seq) > 0:
                        predict_seq.append(j)
                elif predicted[i][j] == 'O':
                    if len(predict_seq) > 0:
                        predict_items.append(predict_seq)
                        predict_seq = []
            if len(golden_seq) > 0:
                golden_items.append(golden_seq)
            if len(predict_seq) > 0:
                predict_items.append(predict_seq)
            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = sum([item in golden_items for item in predict_items])
            # print(correct_01)
            # print([item in golden_items for item in predict_items])
            # print(golden_items)
            # print(predict_items)

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def estimate(self, ds: Iterable[Instance]) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            golden_tags = []
            predicted_tags = []
            eval_loss = 0
            nb_batches = 0
            for batch in pred_generator_tqdm:
                batch = allennlp_util.move_to_device(batch, self.cuda_device)
                nb_batches += 1

                eval_output_dict = self.model.forward(**batch)
                eval_output_dict_decoded = self.model.decode(eval_output_dict)

                golden_tags.extend([instance['opinion_words_tags'] for instance in batch['sample']])
                predicted_tags.extend(eval_output_dict_decoded['tags'])

                loss = eval_output_dict["loss"]
                eval_loss += loss.item()
                metrics = self.model.get_metrics()
                metrics["loss"] = float(eval_loss / nb_batches)

        bio_metrics = self.score_BIO(predicted_tags, golden_tags)

        metrics = self.model.get_metrics(reset=True)
        metrics["loss"] = float(eval_loss / nb_batches)
        metrics.update(bio_metrics)
        return metrics


class ToweEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration

    def score_BIO(self, predicted, golden, ignore_index=-1):
        # tag2id = {'B': 1, 'I': 2, 'O': 0}
        assert len(predicted) == len(golden)
        sum_all = 0
        sum_correct = 0
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        # print(predicted)
        # print(golden)
        for i in range(len(golden)):
            length = len(golden[i])
            # print(length)
            # print(predicted[i])
            # print(golden[i])
            golden_01 = 0
            correct_01 = 0
            predict_01 = 0
            predict_items = []
            golden_items = []
            golden_seq = []
            predict_seq = []
            golden_i = golden[i]
            predicted_i = predicted[i]
            for j in range(length):
                if golden[i][j] == ignore_index:
                    break
                if golden[i][j] == 'B':
                    if len(golden_seq) > 0:  # 00
                        golden_items.append(golden_seq)
                        golden_seq = []
                    golden_seq.append(j)
                elif golden[i][j] == 'I':
                    if len(golden_seq) > 0:
                        golden_seq.append(j)
                elif golden[i][j] == 'O':
                    if len(golden_seq) > 0:
                        golden_items.append(golden_seq)
                        golden_seq = []
                if predicted[i][j] == 'B':
                    if len(predict_seq) > 0:  # 00
                        predict_items.append(predict_seq)
                        predict_seq = []
                    predict_seq.append(j)
                elif predicted[i][j] == 'I':
                    if len(predict_seq) > 0:
                        predict_seq.append(j)
                elif predicted[i][j] == 'O':
                    if len(predict_seq) > 0:
                        predict_items.append(predict_seq)
                        predict_seq = []
            if len(golden_seq) > 0:
                golden_items.append(golden_seq)
            if len(predict_seq) > 0:
                predict_items.append(predict_seq)
            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = sum([item in golden_items for item in predict_items])
            # print(correct_01)
            # print([item in golden_items for item in predict_items])
            # print(golden_items)
            # print(predict_items)

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def first_term_from_tags(self, tags: List[str], start_index: int):
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

    def terms_from_tags(self, tags: List[str], words: List[str]):
        """

        :param tags:
        :return:
        """
        tags = tags[: len(words)]

        terms = []
        start_index = 0
        while start_index < len(tags):
            term = self.first_term_from_tags(tags, start_index)
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

    def precision_recall_f1(self, pred: set, true: set):
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

    def gold_text_aspect_opinion_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s' % (text, aspect_term, opinion_term)
                    result.append(item)
        return result

    def text_aspect_opinion_for_estimation(self, samples: List[dict], tags: List[List[str]]):
        result = {}
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_words_tags = tags[i]
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            result['%s-%s' % (text, aspect_terms[0])] = opinion_terms
        return result

    def estimate_aspect_term_opinin_term_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                              predicted_tags: List[List[str]]):
        gold_text_aspect_opinions = self.gold_text_aspect_opinion_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        all_pred_text_aspect_opinions = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinions.append('%s-%s' % (text_aspect_term, opinion_term))
        return self.precision_recall_f1(set(all_pred_text_aspect_opinions),
                                                         set(gold_text_aspect_opinions))

    def estimate(self, ds: Iterable[Instance]) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))

            samples = []
            golden_tags = []
            predicted_tags = []
            eval_loss = 0
            nb_batches = 0
            for batch in pred_generator_tqdm:
                samples.extend(batch['sample'])

                batch = allennlp_util.move_to_device(batch, self.cuda_device)
                nb_batches += 1

                eval_output_dict = self.model.forward(**batch)
                eval_output_dict_decoded = self.model.decode(eval_output_dict)

                golden_tags.extend([instance['opinion_words_tags'] for instance in batch['sample']])
                predicted_tags.extend(eval_output_dict_decoded['tags'])

                loss = eval_output_dict["loss"]
                eval_loss += loss.item()
                metrics = self.model.get_metrics()
                metrics["loss"] = float(eval_loss / nb_batches)

        bio_metrics = self.score_BIO(predicted_tags, golden_tags)

        metrics = self.model.get_metrics(reset=True)
        metrics["loss"] = float(eval_loss / nb_batches)
        metrics.update(bio_metrics)

        ate_result_filepath = self.configuration['ate_result_filepath']
        if ate_result_filepath:
            ate_result = file_utils.read_all_lines(ate_result_filepath)
            text_and_ate_pred = {}
            for line in ate_result:
                line_dict = json.loads(line, encoding='utf-8')
                ate_pred = line_dict['pred']
                if self.configuration['model_name'] in ['TermBiLSTM', 'TermBert']:
                    adjusted_ate_pred = []
                    for aspect_term in ate_pred:
                        parts = aspect_term.split('-')
                        start_index = str(int(parts[-2]) + 1)
                        end_index = str(int(parts[-1]) + 1)
                        parts[-2] = start_index
                        parts[-1] = end_index
                        adjusted_ate_pred.append('-'.join(parts))
                else:
                    adjusted_ate_pred = ate_pred
                text_and_ate_pred[line_dict['text']] = adjusted_ate_pred

            # aspect term, opinion term pair evaluation
            aspect_term_opinion_term_metrics = self.estimate_aspect_term_opinin_term_pair(samples, text_and_ate_pred,
                                                                                          predicted_tags)
            metrics['aspect_term_opinion_term_metrics'] = aspect_term_opinion_term_metrics

        return metrics


class AsteEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration
        self._accuracy = metrics.CategoricalAccuracy()

    def score_BIO(self, predicted, golden, ignore_index=-1):
        # tag2id = {'B': 1, 'I': 2, 'O': 0}
        assert len(predicted) == len(golden)
        sum_all = 0
        sum_correct = 0
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        # print(predicted)
        # print(golden)
        for i in range(len(golden)):
            length = len(golden[i])
            # print(length)
            # print(predicted[i])
            # print(golden[i])
            golden_01 = 0
            correct_01 = 0
            predict_01 = 0
            predict_items = []
            golden_items = []
            golden_seq = []
            predict_seq = []
            golden_i = golden[i]
            predicted_i = predicted[i]
            for j in range(length):
                if golden[i][j] == ignore_index:
                    break
                if golden[i][j] == 'B':
                    if len(golden_seq) > 0:  # 00
                        golden_items.append(golden_seq)
                        golden_seq = []
                    golden_seq.append(j)
                elif golden[i][j] == 'I':
                    if len(golden_seq) > 0:
                        golden_seq.append(j)
                elif golden[i][j] == 'O':
                    if len(golden_seq) > 0:
                        golden_items.append(golden_seq)
                        golden_seq = []
                if predicted[i][j] == 'B':
                    if len(predict_seq) > 0:  # 00
                        predict_items.append(predict_seq)
                        predict_seq = []
                    predict_seq.append(j)
                elif predicted[i][j] == 'I':
                    if len(predict_seq) > 0:
                        predict_seq.append(j)
                elif predicted[i][j] == 'O':
                    if len(predict_seq) > 0:
                        predict_items.append(predict_seq)
                        predict_seq = []
            if len(golden_seq) > 0:
                golden_items.append(golden_seq)
            if len(predict_seq) > 0:
                predict_items.append(predict_seq)
            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = sum([item in golden_items for item in predict_items])
            # print(correct_01)
            # print([item in golden_items for item in predict_items])
            # print(golden_items)
            # print(predict_items)

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def first_term_from_tags(self, tags: List[str], start_index: int):
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

    def terms_from_tags(self, tags: List[str], words: List[str]):
        """

        :param tags:
        :return:
        """
        tags = tags[: len(words)]

        terms = []
        start_index = 0
        while start_index < len(tags):
            term = self.first_term_from_tags(tags, start_index)
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

    def precision_recall_f1(self, pred: set, true: set):
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

    def gold_text_aspect_opinion_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s' % (text, aspect_term, opinion_term)
                    result.append(item)
        return result

    def text_aspect_opinion_for_estimation(self, samples: List[dict], tags: List[List[str]]):
        result = {}
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_words_tags = tags[i]
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            result['%s-%s' % (text, aspect_terms[0])] = opinion_terms
        return result

    def estimate_aspect_term_opinin_term_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                              predicted_tags: List[List[str]]):
        gold_text_aspect_opinions = self.gold_text_aspect_opinion_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        all_pred_text_aspect_opinions = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinions.append('%s-%s' % (text_aspect_term, opinion_term))
        return self.precision_recall_f1(set(all_pred_text_aspect_opinions),
                                                         set(gold_text_aspect_opinions))

    def gold_text_aspect_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            sentiment = sample['polarity']
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            item = '%s-%s-%s' % (text, aspect_terms[0], sentiment)
            result.append(item)
        return result

    def text_aspect_sentiment_for_estimation(self, samples: List[dict], sentiment_logit_total):
        result = {}
        sentiment_logit_total_list = sentiment_logit_total.detach().cpu().numpy().tolist()
        polarities = self.configuration['polarities'].split(',')
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            sentiment_logit: List = sentiment_logit_total_list[i]
            sentiment_index = sentiment_logit.index(max(sentiment_logit))
            sentiment = polarities[sentiment_index]
            result['%s-%s' % (text, aspect_terms[0])] = sentiment
        return result

    def estimate_aspect_term_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                            sentiment_logit_total):
        gold_text_aspect_sentiments = self.gold_text_aspect_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                all_pred_text_aspect_sentiments.append('%s-%s' % (text_aspect_term, sentiment))
        return self.precision_recall_f1(set(all_pred_text_aspect_sentiments),
                                                         set(gold_text_aspect_sentiments))

    def gold_text_aspect_opinion_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            sentiment = sample['polarity']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s-%s' % (text, aspect_term, opinion_term, sentiment)
                    result.append(item)
        return result

    def gold_text_aspect_for_estimation_from_samples(self, samples: List[dict]):
        result = {}
        for sample in samples:
            words = sample['words']
            # text = sample['metadata']['original_line'].split('####')[0]
            text = sample['metadata']['original_line_data']['sentence']
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if text not in result:
                result[text] = []
            result[text].append(aspect_terms[0])
        return result

    def estimate_aspect_opinion_sentiment_triplet(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def estimate_opinion_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        text_and_ate_pred = self.gold_text_aspect_for_estimation_from_samples(samples)
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def estimate(self, ds: Iterable[Instance], data_type=None) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            samples = []

            golden_tags = []
            predicted_tags = []
            eval_loss = 0
            nb_batches = 0

            sentiment_logits = []
            sentiment_labels = []
            for batch in pred_generator_tqdm:
                samples.extend(batch['sample'])

                batch = allennlp_util.move_to_device(batch, self.cuda_device)
                nb_batches += 1

                eval_output_dict = self.model.forward(**batch)

                # towe
                towe_result = eval_output_dict['towe_result']

                towe_result_dict_decoded = self.model.decode(towe_result)

                golden_tags.extend([instance['opinion_words_tags'] for instance in batch['sample']])
                predicted_tags.extend(towe_result_dict_decoded['tags'])

                loss = towe_result["loss"]
                eval_loss += loss.item()
                metrics = self.model.get_metrics()
                metrics["loss"] = float(eval_loss / nb_batches)

                # atsa
                atsa_result = eval_output_dict['atsa_result']
                sentiment_logit = atsa_result['logit']
                sentiment_label = atsa_result['label']
                sentiment_logits.append(sentiment_logit)
                sentiment_labels.append(sentiment_label)

        bio_metrics = self.score_BIO(predicted_tags, golden_tags)

        metrics = self.model.get_metrics(reset=True)
        metrics["loss"] = float(eval_loss / nb_batches)
        metrics.update(bio_metrics)

        sentiment_logit_total = torch.cat(sentiment_logits, dim=0)
        sentiment_label_total = torch.cat(sentiment_labels, dim=0)
        self._accuracy(sentiment_logit_total, sentiment_label_total)
        sentiment_acc = self._accuracy.get_metric(reset=True)
        metrics['sentiment_acc'] = sentiment_acc

        # evaluate opinion term sentiment pair
        opinion_sentiment_metrics = self.estimate_opinion_sentiment_pair(samples, None,
                                                                      predicted_tags,
                                                                      sentiment_logit_total)
        metrics['opinion_sentiment_metrics'] = opinion_sentiment_metrics
        metrics['opinion_sentiment_f1'] = opinion_sentiment_metrics['f1']

        ate_result_filepath = self.configuration['ate_result_filepath']
        if ate_result_filepath:
            ate_result = file_utils.read_all_lines(ate_result_filepath)
            text_and_ate_pred = {}
            for line in ate_result:
                line_dict = json.loads(line, encoding='utf-8')
                ate_pred = line_dict['pred']
                if self.configuration['model_name'] in ['AsteTermBiLSTM', 'AsteTermBert'] \
                        and self.configuration['aspect_term_aware']:
                    adjusted_ate_pred = []
                    for aspect_term in ate_pred:
                        parts = aspect_term.split('-')
                        start_index = str(int(parts[-2]) + 1)
                        end_index = str(int(parts[-1]) + 1)
                        parts[-2] = start_index
                        parts[-1] = end_index
                        adjusted_ate_pred.append('-'.join(parts))
                text_and_ate_pred[line_dict['text']] = adjusted_ate_pred

            # aspect term, opinion term pair evaluation
            aspect_term_opinion_term_metrics = self.estimate_aspect_term_opinin_term_pair(samples, text_and_ate_pred,
                                                                                          predicted_tags)
            metrics['aspect_term_opinion_term_metrics'] = aspect_term_opinion_term_metrics
            metrics['aspect_term_opinion_term_f1'] = aspect_term_opinion_term_metrics['f1']

            # aspect term, sentiment pair evaluation
            aspect_term_sentiment_pair_metrics = self.estimate_aspect_term_sentiment_pair(samples, text_and_ate_pred,
                                                                                          sentiment_logit_total)
            metrics['aspect_term_sentiment_pair_metrics'] = aspect_term_sentiment_pair_metrics
            metrics['aspect_term_sentiment_pair_f1'] = aspect_term_sentiment_pair_metrics['f1']

            # aste evaluation
            aste_metrics = self.estimate_aspect_opinion_sentiment_triplet(samples, text_and_ate_pred,
                                                                          predicted_tags,
                                                                          sentiment_logit_total)
            metrics['aste_metrics'] = aste_metrics
            metrics['aste_f1'] = aste_metrics['f1']

        return metrics


class MilAsoEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration
        self._accuracy = metrics.CategoricalAccuracy()

    def score_BIO(self, predicted, golden, ignore_index=-1):
        # tag2id = {'B': 1, 'I': 2, 'O': 0}
        assert len(predicted) == len(golden)
        sum_all = 0
        sum_correct = 0
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        # print(predicted)
        # print(golden)
        for i in range(len(golden)):
            length = len(golden[i])
            # print(length)
            # print(predicted[i])
            # print(golden[i])
            golden_01 = 0
            correct_01 = 0
            predict_01 = 0
            predict_items = []
            golden_items = []
            golden_seq = []
            predict_seq = []
            golden_i = golden[i]
            predicted_i = predicted[i]
            for j in range(length):
                if golden[i][j] == ignore_index:
                    break
                if golden[i][j] == 'B':
                    if len(golden_seq) > 0:  # 00
                        golden_items.append(golden_seq)
                        golden_seq = []
                    golden_seq.append(j)
                elif golden[i][j] == 'I':
                    if len(golden_seq) > 0:
                        golden_seq.append(j)
                elif golden[i][j] == 'O':
                    if len(golden_seq) > 0:
                        golden_items.append(golden_seq)
                        golden_seq = []
                if predicted[i][j] == 'B':
                    if len(predict_seq) > 0:  # 00
                        predict_items.append(predict_seq)
                        predict_seq = []
                    predict_seq.append(j)
                elif predicted[i][j] == 'I':
                    if len(predict_seq) > 0:
                        predict_seq.append(j)
                elif predicted[i][j] == 'O':
                    if len(predict_seq) > 0:
                        predict_items.append(predict_seq)
                        predict_seq = []
            if len(golden_seq) > 0:
                golden_items.append(golden_seq)
            if len(predict_seq) > 0:
                predict_items.append(predict_seq)
            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = sum([item in golden_items for item in predict_items])
            # print(correct_01)
            # print([item in golden_items for item in predict_items])
            # print(golden_items)
            # print(predict_items)

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def first_term_from_tags(self, tags: List[str], start_index: int):
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

    def terms_from_tags(self, tags: List[str], words: List[str]):
        """

        :param tags:
        :return:
        """
        tags = tags[: len(words)]

        terms = []
        start_index = 0
        while start_index < len(tags):
            term = self.first_term_from_tags(tags, start_index)
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

    def precision_recall_f1(self, pred: set, true: set):
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

    def gold_text_aspect_opinion_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s' % (text, aspect_term, opinion_term)
                    result.append(item)
        return result

    def text_aspect_opinion_for_estimation(self, samples: List[dict], tags: List[List[str]]):
        result = {}
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_words_tags = tags[i]
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            result['%s-%s' % (text, aspect_terms[0])] = opinion_terms
        return result

    def estimate_aspect_term_opinin_term_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                              predicted_tags: List[List[str]]):
        gold_text_aspect_opinions = self.gold_text_aspect_opinion_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        all_pred_text_aspect_opinions = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinions.append('%s-%s' % (text_aspect_term, opinion_term))
        return self.precision_recall_f1(set(all_pred_text_aspect_opinions),
                                                         set(gold_text_aspect_opinions))

    def gold_text_aspect_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            sentiment = sample['polarity']
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            item = '%s-%s-%s' % (text, aspect_terms[0], sentiment)
            result.append(item)
        return result

    def text_aspect_sentiment_for_estimation(self, samples: List[dict], sentiment_logit_total):
        result = {}
        sentiment_logit_total_list = sentiment_logit_total.detach().cpu().numpy().tolist()
        polarities = self.configuration['polarities'].split(',')
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            sentiment_logit: List = sentiment_logit_total_list[i]
            sentiment_index = sentiment_logit.index(max(sentiment_logit))
            sentiment = polarities[sentiment_index]
            result['%s-%s' % (text, aspect_terms[0])] = sentiment
        return result

    def estimate_aspect_term_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                            sentiment_logit_total):
        gold_text_aspect_sentiments = self.gold_text_aspect_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                all_pred_text_aspect_sentiments.append('%s-%s' % (text_aspect_term, sentiment))
        return self.precision_recall_f1(set(all_pred_text_aspect_sentiments),
                                                         set(gold_text_aspect_sentiments))

    def gold_text_aspect_opinion_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            sentiment = sample['polarity']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s-%s' % (text, aspect_term, opinion_term, sentiment)
                    result.append(item)
        return result

    def gold_text_aspect_for_estimation_from_samples(self, samples: List[dict]):
        result = {}
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if text not in result:
                result[text] = []
            result[text].append(aspect_terms[0])
        return result

    def estimate_aspect_opinion_sentiment_triplet(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def estimate_opinion_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        text_and_ate_pred = self.gold_text_aspect_for_estimation_from_samples(samples)
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def get_polarity_from_tag(self, tag: str):
        return tag[: tag.index('-')]

    def terms_from_bio_tags(self, tags: List[str]):
        result = []
        start = -1
        polarity = ''
        for i, tag in enumerate(tags):
            if start == -1:
                if 'B' in tag:
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
            else:
                if 'O' == tag:
                    term = '%s-%d-%d' % (polarity, start, i)
                    start = -1
                    polarity = ''
                    result.append(term)
                if 'B' in tag:
                    term = '%s-%d-%d' % (polarity, start, i)
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
                    result.append(term)
        if start != -1:
            term = '%s-%d-%d' % (polarity, start, len(tags))
            result.append(term)
        return result

    def score_BIO_with_polarity(self, predicted, golden, ignore_index=-1):
        assert len(predicted) == len(golden)
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        for i in range(len(golden)):
            length = len(golden[i])
            golden_i = golden[i]
            predicted_i = predicted[i][: length]

            golden_items = self.terms_from_bio_tags(golden_i)
            predict_items = self.terms_from_bio_tags(predicted_i)

            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = 0
            for term in predict_items:
                if term in golden_items:
                    correct_01 += 1

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def term_polarity(self, start, end, sentiment_outputs_of_words_i, towe_attention_i):
        sentiment_outputs_of_words_i_slice = sentiment_outputs_of_words_i[start: end].detach().cpu().numpy()
        if start + 1 == end:
            result = np.argmax(sentiment_outputs_of_words_i_slice, axis=-1)[0]
        else:
            towe_attention_i_slice = towe_attention_i[start: end]
            towe_attention_i_slice_normalized = towe_attention_i_slice.unsqueeze(dim=-1).detach().cpu().numpy()
            sentiment_distribution = sentiment_outputs_of_words_i_slice * towe_attention_i_slice_normalized
            sentiment_distribution_merged = np.sum(sentiment_distribution, axis=0)
            result = np.argmax(sentiment_distribution_merged, axis=-1)
        result = int(result)
        return result

    def tags_with_polarity(self, towe_tags, sentiment_outputs_of_words, towe_attention):
        polarities = self.configuration['polarities'].split(',')
        result = []
        for i in range(len(towe_tags)):
            towe_tags_i = towe_tags[i]
            terms = sequence_labeling_utils.terms_from_tags(towe_tags_i, ['word' for _ in towe_tags_i])
            result_e = ['O' for _ in towe_tags_i]
            for term in terms:
                term_parts = term.split('-')
                term_start = int(term_parts[1])
                term_end = int(term_parts[2])
                sentiment_outputs_of_words_i = sentiment_outputs_of_words[i]
                towe_attention_i = towe_attention[i]
                polarity_index = self.term_polarity(term_start, term_end, sentiment_outputs_of_words_i,
                                                    towe_attention_i)
                polarity = polarities[polarity_index]
                for j in range(term_start, term_end):
                    result_e[j] = '%s-%s' % (polarity, towe_tags_i[j])
            result.append(result_e)

        # sentiment_outputs_of_words = torch.argmax(sentiment_outputs_of_words, dim=-1)
        # sentiment_outputs_of_words = sentiment_outputs_of_words.detach().cpu().numpy().tolist()
        # sentiment_outputs_of_words = [[polarities[ee] for ee in e] for e in sentiment_outputs_of_words]
        # result = []
        # for i in range(len(towe_tags)):
        #     towe_tags_i = towe_tags[i]
        #     sentiment_outputs_of_words_i = sentiment_outputs_of_words[i]
        #     result_e = []
        #     for j in range(len(towe_tags_i)):
        #         if towe_tags_i[j] == 'O':
        #             result_e.append('O')
        #         else:
        #             result_e.append('%s-%s' % (sentiment_outputs_of_words_i[j], towe_tags_i[j]))
        #     result.append(result_e)
        return result

    def multilabel_accuracy(self, logits, labels):
        pred = torch.argmax(logits, dim=-1)
        true = torch.argmax(labels, dim=-1)
        acc = accuracy_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
        return {'sentiment_accuracy': acc}

    def estimate(self, ds: Iterable[Instance], data_type=None) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            samples = []

            golden_tags = []
            predicted_tags = []
            eval_loss = 0
            nb_batches = 0

            sentiment_logits = []
            sentiment_labels = []

            golden_tags_with_polarity = []
            predicted_tags_with_polarity = []
            for batch in pred_generator_tqdm:
                samples.extend(batch['sample'])

                batch = allennlp_util.move_to_device(batch, self.cuda_device)
                nb_batches += 1

                eval_output_dict = self.model.forward(**batch)

                loss = eval_output_dict["loss"]
                eval_loss += loss.item()

                # towe
                towe_result = eval_output_dict['towe_result']

                towe_result_dict_decoded = self.model.decode(towe_result)

                golden_tags.extend([instance['opinion_words_tags'] for instance in batch['sample']])
                predicted_tags.extend(towe_result_dict_decoded['tags'])

                golden_tags_with_polarity.extend([instance['opinion_words_tags_with_polarity'] for instance in batch['sample']])

                # atsa
                atsa_result = eval_output_dict['atsa_result']
                sentiment_logit = atsa_result['logit']
                sentiment_label = atsa_result['label']
                sentiment_logits.append(sentiment_logit)
                sentiment_labels.append(sentiment_label)

                sentiment_outputs_of_words = torch.softmax(atsa_result['sentiment_outputs_of_words'], dim=-1)
                towe_attention = atsa_result['towe_attention']
                predicted_tags_with_polarity.extend(self.tags_with_polarity(towe_result_dict_decoded['tags'],
                                                                            sentiment_outputs_of_words,
                                                                            towe_attention))

            bio_metrics = self.score_BIO(predicted_tags, golden_tags)

            bio_with_polarity_metrics = self.score_BIO_with_polarity(predicted_tags_with_polarity,
                                                                     golden_tags_with_polarity)

            metrics = {}
            metrics["loss"] = float(eval_loss / nb_batches)
            metrics['towe'] = bio_metrics
            metrics['so'] = bio_with_polarity_metrics

            sentiment_logit_total = torch.cat(sentiment_logits, dim=0)
            sentiment_label_total = torch.cat(sentiment_labels, dim=0)
            sentiment_accuracy = self.multilabel_accuracy(sentiment_logit_total, sentiment_label_total)
            metrics['sentiment_accuracy'] = sentiment_accuracy

            metrics['sentiment_acc'] = sentiment_accuracy['sentiment_accuracy']
            metrics['f1'] = bio_metrics['f1']
            metrics['opinion_sentiment_f1'] = bio_with_polarity_metrics['f1']

        return metrics


class AsoEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration
        self._accuracy = metrics.CategoricalAccuracy()

    def get_polarity_from_tag(self, tag: str):
        return tag[: tag.index('-')]

    def terms_from_bio_tags(self, tags: List[str]):
        result = []
        start = -1
        polarity = ''
        for i, tag in enumerate(tags):
            if start == -1:
                if 'B' in tag:
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
            else:
                if 'O' == tag:
                    term = '%s-%d-%d' % (polarity, start, i)
                    start = -1
                    polarity = ''
                    result.append(term)
                if 'B' in tag:
                    term = '%s-%d-%d' % (polarity, start, i)
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
                    result.append(term)
        if start != -1:
            term = '%s-%d-%d' % (polarity, start, len(tags))
            result.append(term)
        return result

    def score_BIO(self, predicted, golden, ignore_index=-1):
        assert len(predicted) == len(golden)
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        for i in range(len(golden)):
            length = len(golden[i])
            golden_i = golden[i]
            predicted_i = predicted[i][: length]

            golden_items = self.terms_from_bio_tags(golden_i)
            predict_items = self.terms_from_bio_tags(predicted_i)

            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = 0
            for term in predict_items:
                if term in golden_items:
                    correct_01 += 1

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def first_term_from_tags(self, tags: List[str], start_index: int):
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

    def terms_from_tags(self, tags: List[str], words: List[str]):
        """

        :param tags:
        :return:
        """
        tags = tags[: len(words)]

        terms = []
        start_index = 0
        while start_index < len(tags):
            term = self.first_term_from_tags(tags, start_index)
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

    def precision_recall_f1(self, pred: set, true: set):
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

    def gold_text_aspect_opinion_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s' % (text, aspect_term, opinion_term)
                    result.append(item)
        return result

    def text_aspect_opinion_for_estimation(self, samples: List[dict], tags: List[List[str]]):
        result = {}
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_words_tags = tags[i]
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            result['%s-%s' % (text, aspect_terms[0])] = opinion_terms
        return result

    def estimate_aspect_term_opinin_term_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                              predicted_tags: List[List[str]]):
        gold_text_aspect_opinions = self.gold_text_aspect_opinion_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        all_pred_text_aspect_opinions = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinions.append('%s-%s' % (text_aspect_term, opinion_term))
        return self.precision_recall_f1(set(all_pred_text_aspect_opinions),
                                                         set(gold_text_aspect_opinions))

    def gold_text_aspect_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            sentiment = sample['polarity']
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            item = '%s-%s-%s' % (text, aspect_terms[0], sentiment)
            result.append(item)
        return result

    def text_aspect_sentiment_for_estimation(self, samples: List[dict], sentiment_logit_total):
        result = {}
        sentiment_logit_total_list = sentiment_logit_total.detach().cpu().numpy().tolist()
        polarities = self.configuration['polarities'].split(',')
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            sentiment_logit: List = sentiment_logit_total_list[i]
            sentiment_index = sentiment_logit.index(max(sentiment_logit))
            sentiment = polarities[sentiment_index]
            result['%s-%s' % (text, aspect_terms[0])] = sentiment
        return result

    def estimate_aspect_term_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                            sentiment_logit_total):
        gold_text_aspect_sentiments = self.gold_text_aspect_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                all_pred_text_aspect_sentiments.append('%s-%s' % (text_aspect_term, sentiment))
        return self.precision_recall_f1(set(all_pred_text_aspect_sentiments),
                                                         set(gold_text_aspect_sentiments))

    def gold_text_aspect_opinion_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            sentiment = sample['polarity']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s-%s' % (text, aspect_term, opinion_term, sentiment)
                    result.append(item)
        return result

    def gold_text_aspect_for_estimation_from_samples(self, samples: List[dict]):
        result = {}
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if text not in result:
                result[text] = []
            result[text].append(aspect_terms[0])
        return result

    def estimate_aspect_opinion_sentiment_triplet(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def estimate_opinion_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        text_and_ate_pred = self.gold_text_aspect_for_estimation_from_samples(samples)
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def estimate(self, ds: Iterable[Instance], data_type=None) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            samples = []

            golden_tags = []
            predicted_tags = []
            eval_loss = 0
            nb_batches = 0

            for batch in pred_generator_tqdm:
                samples.extend(batch['sample'])

                batch = allennlp_util.move_to_device(batch, self.cuda_device)
                nb_batches += 1

                eval_output_dict = self.model.forward(**batch)

                so_result_dict_decoded = self.model.decode(eval_output_dict)

                golden_tags.extend([instance['opinion_words_tags'] for instance in batch['sample']])
                predicted_tags.extend(so_result_dict_decoded['tags'])

                loss = eval_output_dict["loss"]
                eval_loss += loss.item()

            bio_metrics = self.score_BIO(predicted_tags, golden_tags)

            metrics = self.model.get_metrics(reset=True)
            metrics["loss"] = float(eval_loss / nb_batches)
            metrics['bio_metrics'] = bio_metrics
            metrics['opinion_sentiment_f1'] = bio_metrics['f1']

        return metrics


class Predictor:

    def predict(self, ds: Iterable[Instance]) -> dict:
        raise NotImplementedError('predict')


class MilAsoPredictor(Predictor):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration
        self._accuracy = metrics.CategoricalAccuracy()

    def score_BIO(self, predicted, golden, ignore_index=-1):
        # tag2id = {'B': 1, 'I': 2, 'O': 0}
        assert len(predicted) == len(golden)
        sum_all = 0
        sum_correct = 0
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        # print(predicted)
        # print(golden)
        for i in range(len(golden)):
            length = len(golden[i])
            # print(length)
            # print(predicted[i])
            # print(golden[i])
            golden_01 = 0
            correct_01 = 0
            predict_01 = 0
            predict_items = []
            golden_items = []
            golden_seq = []
            predict_seq = []
            golden_i = golden[i]
            predicted_i = predicted[i]
            for j in range(length):
                if golden[i][j] == ignore_index:
                    break
                if golden[i][j] == 'B':
                    if len(golden_seq) > 0:  # 00
                        golden_items.append(golden_seq)
                        golden_seq = []
                    golden_seq.append(j)
                elif golden[i][j] == 'I':
                    if len(golden_seq) > 0:
                        golden_seq.append(j)
                elif golden[i][j] == 'O':
                    if len(golden_seq) > 0:
                        golden_items.append(golden_seq)
                        golden_seq = []
                if predicted[i][j] == 'B':
                    if len(predict_seq) > 0:  # 00
                        predict_items.append(predict_seq)
                        predict_seq = []
                    predict_seq.append(j)
                elif predicted[i][j] == 'I':
                    if len(predict_seq) > 0:
                        predict_seq.append(j)
                elif predicted[i][j] == 'O':
                    if len(predict_seq) > 0:
                        predict_items.append(predict_seq)
                        predict_seq = []
            if len(golden_seq) > 0:
                golden_items.append(golden_seq)
            if len(predict_seq) > 0:
                predict_items.append(predict_seq)
            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = sum([item in golden_items for item in predict_items])
            # print(correct_01)
            # print([item in golden_items for item in predict_items])
            # print(golden_items)
            # print(predict_items)

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def first_term_from_tags(self, tags: List[str], start_index: int):
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

    def terms_from_tags(self, tags: List[str], words: List[str]):
        """

        :param tags:
        :return:
        """
        tags = tags[: len(words)]

        terms = []
        start_index = 0
        while start_index < len(tags):
            term = self.first_term_from_tags(tags, start_index)
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

    def precision_recall_f1(self, pred: set, true: set):
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

    def gold_text_aspect_opinion_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s' % (text, aspect_term, opinion_term)
                    result.append(item)
        return result

    def text_aspect_opinion_for_estimation(self, samples: List[dict], tags: List[List[str]]):
        result = {}
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_words_tags = tags[i]
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            result['%s-%s' % (text, aspect_terms[0])] = opinion_terms
        return result

    def estimate_aspect_term_opinin_term_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                              predicted_tags: List[List[str]]):
        gold_text_aspect_opinions = self.gold_text_aspect_opinion_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        all_pred_text_aspect_opinions = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinions.append('%s-%s' % (text_aspect_term, opinion_term))
        return self.precision_recall_f1(set(all_pred_text_aspect_opinions),
                                                         set(gold_text_aspect_opinions))

    def gold_text_aspect_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            sentiment = sample['polarity']
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            item = '%s-%s-%s' % (text, aspect_terms[0], sentiment)
            result.append(item)
        return result

    def text_aspect_sentiment_for_estimation(self, samples: List[dict], sentiment_logit_total):
        result = {}
        sentiment_logit_total_list = sentiment_logit_total.detach().cpu().numpy().tolist()
        polarities = self.configuration['polarities'].split(',')
        for i in range(len(samples)):
            sample = samples[i]
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            sentiment_logit: List = sentiment_logit_total_list[i]
            sentiment_index = sentiment_logit.index(max(sentiment_logit))
            sentiment = polarities[sentiment_index]
            result['%s-%s' % (text, aspect_terms[0])] = sentiment
        return result

    def estimate_aspect_term_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                            sentiment_logit_total):
        gold_text_aspect_sentiments = self.gold_text_aspect_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                all_pred_text_aspect_sentiments.append('%s-%s' % (text_aspect_term, sentiment))
        return self.precision_recall_f1(set(all_pred_text_aspect_sentiments),
                                                         set(gold_text_aspect_sentiments))

    def gold_text_aspect_opinion_sentiment_for_estimation_from_samples(self, samples: List[dict]):
        result = []
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            opinion_words_tags = sample['opinion_words_tags']
            sentiment = sample['polarity']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if len(aspect_terms) > 1:
                raise Exception('size of aspect_terms > 1')
            opinion_terms = self.terms_from_tags(opinion_words_tags, words)
            if len(opinion_terms) == 0:
                opinion_terms = ['-']
            for aspect_term in aspect_terms:
                for opinion_term in opinion_terms:
                    item = '%s-%s-%s-%s' % (text, aspect_term, opinion_term, sentiment)
                    result.append(item)
        return result

    def gold_text_aspect_for_estimation_from_samples(self, samples: List[dict]):
        result = {}
        for sample in samples:
            words = sample['words']
            text = sample['metadata']['original_line'].split('####')[0]
            target_tags = sample['target_tags']
            aspect_terms = self.terms_from_tags(target_tags, words)
            if text not in result:
                result[text] = []
            result[text].append(aspect_terms[0])
        return result

    def estimate_aspect_opinion_sentiment_triplet(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def estimate_opinion_sentiment_pair(self, samples: List[dict], text_and_ate_pred: dict,
                                                  predicted_tags: List[List[str]],
                                                  sentiment_logit_total):
        gold_text_aspect_opinion_sentiments = self.gold_text_aspect_opinion_sentiment_for_estimation_from_samples(samples)
        pred_text_aspect_opinions = self.text_aspect_opinion_for_estimation(samples, predicted_tags)
        pred_text_aspect_sentiments = self.text_aspect_sentiment_for_estimation(samples, sentiment_logit_total)
        all_pred_text_aspect_opinion_sentiments = []
        text_and_ate_pred = self.gold_text_aspect_for_estimation_from_samples(samples)
        for text, aspect_terms in text_and_ate_pred.items():
            for aspect_term in aspect_terms:
                text_aspect_term = '%s-%s' % (text, aspect_term)
                opinion_terms = ['-']
                if text_aspect_term in pred_text_aspect_opinions:
                    opinion_terms = pred_text_aspect_opinions[text_aspect_term]
                sentiment = '-'
                if text_aspect_term in pred_text_aspect_sentiments:
                    sentiment = pred_text_aspect_sentiments[text_aspect_term]
                for opinion_term in opinion_terms:
                    all_pred_text_aspect_opinion_sentiments.append('%s-%s-%s' % (text_aspect_term, opinion_term,
                                                                                 sentiment))
        return self.precision_recall_f1(set(gold_text_aspect_opinion_sentiments),
                                                         set(all_pred_text_aspect_opinion_sentiments))

    def get_polarity_from_tag(self, tag: str):
        return tag[: tag.index('-')]

    def terms_from_bio_tags(self, tags: List[str]):
        result = []
        start = -1
        polarity = ''
        for i, tag in enumerate(tags):
            if start == -1:
                if 'B' in tag:
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
            else:
                if 'O' == tag:
                    term = '%s-%d-%d' % (polarity, start, i)
                    start = -1
                    polarity = ''
                    result.append(term)
                if 'B' in tag:
                    term = '%s-%d-%d' % (polarity, start, i)
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
                    result.append(term)
        if start != -1:
            term = '%s-%d-%d' % (polarity, start, len(tags))
            result.append(term)
        return result

    def score_BIO_with_polarity(self, predicted, golden, ignore_index=-1):
        assert len(predicted) == len(golden)
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        for i in range(len(golden)):
            length = len(golden[i])
            golden_i = golden[i]
            predicted_i = predicted[i][: length]

            golden_items = self.terms_from_bio_tags(golden_i)
            predict_items = self.terms_from_bio_tags(predicted_i)

            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = 0
            for term in predict_items:
                if term in golden_items:
                    correct_01 += 1

            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count / predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count / golden_01_count if golden_01_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        score_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        return score_dict

    def term_polarity(self, start, end, sentiment_outputs_of_words_i, towe_attention_i):
        sentiment_outputs_of_words_i_slice = sentiment_outputs_of_words_i[start: end].detach().cpu().numpy()
        if start + 1 == end:
            result = np.argmax(sentiment_outputs_of_words_i_slice, axis=-1)[0]
        else:
            towe_attention_i_slice = towe_attention_i[start: end]
            towe_attention_i_slice_normalized = towe_attention_i_slice.unsqueeze(dim=-1).detach().cpu().numpy()
            sentiment_distribution = sentiment_outputs_of_words_i_slice * towe_attention_i_slice_normalized
            sentiment_distribution_merged = np.sum(sentiment_distribution, axis=0)
            result = np.argmax(sentiment_distribution_merged, axis=-1)
        result = int(result)
        return result

    def tags_with_polarity(self, towe_tags, sentiment_outputs_of_words, towe_attention):
        polarities = self.configuration['polarities'].split(',')
        result = []
        for i in range(len(towe_tags)):
            towe_tags_i = towe_tags[i]
            terms = sequence_labeling_utils.terms_from_tags(towe_tags_i, ['word' for _ in towe_tags_i])
            result_e = ['O' for _ in towe_tags_i]
            for term in terms:
                term_parts = term.split('-')
                term_start = int(term_parts[1])
                term_end = int(term_parts[2])
                sentiment_outputs_of_words_i = sentiment_outputs_of_words[i]
                towe_attention_i = towe_attention[i]
                polarity_index = self.term_polarity(term_start, term_end, sentiment_outputs_of_words_i,
                                                    towe_attention_i)
                polarity = polarities[polarity_index]
                for j in range(term_start, term_end):
                    result_e[j] = '%s-%s' % (polarity, towe_tags_i[j])
            result.append(result_e)

        # sentiment_outputs_of_words = torch.argmax(sentiment_outputs_of_words, dim=-1)
        # sentiment_outputs_of_words = sentiment_outputs_of_words.detach().cpu().numpy().tolist()
        # sentiment_outputs_of_words = [[polarities[ee] for ee in e] for e in sentiment_outputs_of_words]
        # result = []
        # for i in range(len(towe_tags)):
        #     towe_tags_i = towe_tags[i]
        #     sentiment_outputs_of_words_i = sentiment_outputs_of_words[i]
        #     result_e = []
        #     for j in range(len(towe_tags_i)):
        #         if towe_tags_i[j] == 'O':
        #             result_e.append('O')
        #         else:
        #             result_e.append('%s-%s' % (sentiment_outputs_of_words_i[j], towe_tags_i[j]))
        #     result.append(result_e)
        return result

    def multilabel_accuracy(self, logits, labels):
        pred = torch.argmax(logits, dim=-1)
        true = torch.argmax(labels, dim=-1)
        acc = accuracy_score(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
        return {'sentiment_accuracy': acc}

    def generate_opinion_term(self, polarity: str, start: int, end: int, words: List[str]):
        """

        :param polarity:
        :param start:
        :param end:
        :param words:
        :return:
        """
        return {'polarity': polarity, 'start': start, 'end': end, 'term': ' '.join(words[start: end])}

    def terms_from_bio_tags_with_words(self, tags: List[str], words: List[str]):
        result = []
        start = -1
        polarity = ''
        for i, tag in enumerate(tags):
            if start == -1:
                if 'B' in tag:
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
            else:
                if 'O' == tag:
                    term = self.generate_opinion_term(polarity, start, i, words)
                    start = -1
                    polarity = ''
                    result.append(term)
                if 'B' in tag:
                    term = self.generate_opinion_term(polarity, start, i, words)
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
                    result.append(term)
        if start != -1:
            term = self.generate_opinion_term(polarity, start, len(tags), words)
            result.append(term)
        return result

    def adjust_indices_of_words(self, opinions: List[dict], aspect_term_indices: List[int]):
        """

        :param opinions:
        :param aspect_term_indices:
        :return:
        """
        for opinion in opinions:
            if opinion['start'] < aspect_term_indices[0]:
                continue
            else:
                opinion['start'] -= 2
                opinion['end'] -= 2

    def print_word_sentiment_and_attention(self, sentiment_outputs_of_words, attentions, sentences):
        """

        :param sentiment_outputs_of_words:
        :param attentions:
        :param sentences:
        :return:
        """
        print()
        sentiment_outputs_of_words = torch.softmax(sentiment_outputs_of_words, dim=-1).detach().cpu().numpy().tolist()
        attentions = attentions.detach().cpu().numpy().tolist()
        for i in range(len(sentences)):
            words = sentences[i]
            sentiment_output_of_words = sentiment_outputs_of_words[i]
            attention = attentions[i]
            output = {
                'words': words,
                'sentiment_output_of_words': sentiment_output_of_words,
                'attention': attention
            }
            print(json.dumps(output))

    def predict(self, ds: Iterable[Instance], data_type=None) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            samples = []

            golden_tags = []
            predicted_tags = []

            sentiment_labels = []
            sentiment_logits = []

            golden_tags_with_polarity = []
            predicted_tags_with_polarity = []

            test_set_aspect_term_polarities = []
            for batch in pred_generator_tqdm:
                samples.extend(batch['sample'])

                batch = allennlp_util.move_to_device(batch, self.cuda_device)

                [test_set_aspect_term_polarities.append(e['polarity']) for e in batch['sample']]

                eval_output_dict = self.model.forward(**batch)

                # self.print_word_sentiment_and_attention(eval_output_dict['atsa_result']['sentiment_outputs_of_words'],
                #                                         eval_output_dict['atsa_result']['towe_attention'],
                #                                         [e['words'] for e in batch['sample']])

                # towe
                towe_result = eval_output_dict['towe_result']

                towe_result_dict_decoded = self.model.decode(towe_result)
                temp = []
                for i in range(len(towe_result_dict_decoded['tags'])):
                    tags_e = towe_result_dict_decoded['tags'][i]
                    words_e = towe_result_dict_decoded['words'][i]
                    temp.append(tags_e[:len(words_e)])
                towe_result_dict_decoded['tags'] = temp

                golden_tags.extend([instance['opinion_words_tags'] for instance in batch['sample']])
                predicted_tags.extend(towe_result_dict_decoded['tags'])

                golden_tags_with_polarity.extend([instance['opinion_words_tags_with_polarity'] for instance in batch['sample']])

                # atsa
                atsa_result = eval_output_dict['atsa_result']
                sentiment_logit = atsa_result['logit']
                sentiment_label = atsa_result['label']
                sentiment_logits.append(sentiment_logit)
                sentiment_labels.append(sentiment_label)

                sentiment_outputs_of_words = torch.softmax(atsa_result['sentiment_outputs_of_words'], dim=-1)
                towe_attention = atsa_result['towe_attention']
                predicted_tags_with_polarity.extend(self.tags_with_polarity(towe_result_dict_decoded['tags'],
                                                                            sentiment_outputs_of_words,
                                                                            towe_attention))

            sentiment_label_total = torch.cat(sentiment_labels, dim=0)
            sentiment_logit_total = torch.cat(sentiment_logits, dim=0)

            opinions_result = []
            predefined_polarities = self.configuration['polarities'].split(',')
            conflict_counter = 0
            for i in range(len(samples)):
                words = samples[i]['words']
                opinions = self.terms_from_bio_tags_with_words(predicted_tags_with_polarity[i], words)
                opinions_true = self.terms_from_bio_tags_with_words(golden_tags_with_polarity[i], words)
                word_indices_of_aspect_terms = samples[i]['word_indices_of_aspect_terms']
                self.adjust_indices_of_words(opinions, word_indices_of_aspect_terms)
                self.adjust_indices_of_words(opinions_true, word_indices_of_aspect_terms)

                aspect_term_sentiment_label = []
                for j in range(len(predefined_polarities)):
                    polarity_indicator = sentiment_label_total[i][j]
                    if polarity_indicator == 1:
                        aspect_term_sentiment_label.append(predefined_polarities[j])
                if len(aspect_term_sentiment_label) > 1:
                    aspect_term_spolarity = 'conflict'
                    conflict_counter += 1
                else:
                    aspect_term_spolarity = aspect_term_sentiment_label[0]
                aspect_term_spolarity_temp = test_set_aspect_term_polarities[i]
                if aspect_term_spolarity_temp != aspect_term_spolarity:
                    print()

                opinions_result.append({'words': samples[i]['words_backup'],
                                        'opinions': opinions,
                                        'opinions_true': opinions_true,
                                        'word_indices_of_aspect_terms': samples[i]['metadata']['original_line_data']['aspect_term'],
                                        'aspect_term_spolarity': aspect_term_spolarity
                                        })

            return opinions_result


class SequenceLabelingModelPredictor(Predictor):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration

    def predict(self, ds: Iterable[Instance]) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            result = []
            for batch in pred_generator_tqdm:
                batch = allennlp_util.move_to_device(batch, self.cuda_device)

                eval_output_dict = self.model.forward(**batch)
                eval_output_dict = self.model.decode(eval_output_dict)
                result.extend(eval_output_dict['tags'])
        return result


class AstePredictor(Predictor):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration
        self._accuracy = metrics.CategoricalAccuracy()

    def first_term_from_tags(self, tags: List[str], start_index: int):
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

    def terms_from_tags(self, tags: List[str], words: List[str]):
        """

        :param tags:
        :return:
        """
        tags = tags[: len(words)]

        terms = []
        start_index = 0
        while start_index < len(tags):
            term = self.first_term_from_tags(tags, start_index)
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

    def predict(self, ds: Iterable[Instance]) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            predicted_tags = []
            sentiment_logits = []
            for batch in pred_generator_tqdm:
                batch = allennlp_util.move_to_device(batch, self.cuda_device)

                eval_output_dict = self.model.forward(**batch)

                # towe
                towe_result = eval_output_dict['towe_result']
                towe_result_dict_decoded = self.model.decode(towe_result)
                predicted_tags.extend(towe_result_dict_decoded['tags'])

                # atsa
                atsa_result = eval_output_dict['atsa_result']
                sentiment_logit = atsa_result['logit']
                sentiment_logits.append(sentiment_logit)

            sentiment_logit_total = torch.cat(sentiment_logits, dim=0)
            sentiment_indices = sentiment_logit_total.argmax(dim=-1).detach().cpu().numpy().tolist()
            sentiment_polarities = []
            polarities = self.configuration['polarities'].split(',')
            for index in sentiment_indices:
                polarity = polarities[index]
                if polarity == 'POS':
                    polarity = 'positive'
                elif polarity == 'NEG':
                    polarity = 'negative'
                elif polarity == 'NEU':
                    polarity = 'neutral'
                sentiment_polarities.append(polarity)
        return {'predicted_tags': predicted_tags, 'sentiment_polarities': sentiment_polarities}


class AsoPredictor(Predictor):
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.metrics = {}
        self.cuda_device = cuda_device
        self.configuration = configuration
        self._accuracy = metrics.CategoricalAccuracy()

    def get_polarity_from_tag(self, tag: str):
        return tag[: tag.index('-')]

    def generate_opinion_term(self, polarity: str, start: int, end: int, words: List[str]):
        """

        :param polarity:
        :param start:
        :param end:
        :param words:
        :return:
        """
        return {'polarity': polarity, 'start': start, 'end': end, 'term': ' '.join(words[start: end])}

    def terms_from_bio_tags(self, tags: List[str], words: List[str]):
        result = []
        start = -1
        polarity = ''
        for i, tag in enumerate(tags):
            if start == -1:
                if 'B' in tag:
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
            else:
                if 'O' == tag:
                    term = self.generate_opinion_term(polarity, start, i, words)
                    start = -1
                    polarity = ''
                    result.append(term)
                if 'B' in tag:
                    term = self.generate_opinion_term(polarity, start, i, words)
                    start = i
                    polarity = self.get_polarity_from_tag(tag)
                    result.append(term)
        if start != -1:
            term = self.generate_opinion_term(polarity, start, len(tags), words)
            result.append(term)
        return result

    def adjust_indices_of_words(self, opinions: List[dict], aspect_term_indices: List[int]):
        """

        :param opinions:
        :param aspect_term_indices:
        :return:
        """
        for opinion in opinions:
            if opinion['start'] < aspect_term_indices[0]:
                continue
            else:
                opinion['start'] -= 2
                opinion['end'] -= 2

    def predict(self, ds: Iterable[Instance]) -> dict:
        with torch.no_grad():
            self.model.eval()
            pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
            pred_generator_tqdm = tqdm(pred_generator,
                                       total=self.iterator.get_num_batches(ds))
            result = []
            for batch in pred_generator_tqdm:
                batch = allennlp_util.move_to_device(batch, self.cuda_device)

                eval_output_dict = self.model.forward(**batch)

                so_result_dict_decoded = self.model.decode(eval_output_dict)
                sample = batch['sample']
                for i in range(len(sample)):
                    original_line_data = sample[i]['metadata']['original_line_data']

                    words = sample[i]['words']
                    word_indices_of_aspect_terms = sample[i]['word_indices_of_aspect_terms']
                    tags = so_result_dict_decoded['tags'][i][: len(words)]
                    opinions = self.terms_from_bio_tags(tags, words)
                    if self.configuration['model_name'] not in ['AsoBertPair', 'AsoBertPairWithPosition']:
                        self.adjust_indices_of_words(opinions, word_indices_of_aspect_terms)
                    if 'opinion_words_tags' in sample[i]:
                        opinions_true = []
                        for e in original_line_data['opinions']:
                            if 'opinion_term' not in e:
                                continue
                            term_temp = e['opinion_term']
                            term_temp['polarity'] = e['polarity']
                            opinions_true.append(term_temp)
                    else:
                        opinions_true = []
                    result.append({'words': original_line_data['words'], 'opinions': opinions, 'opinions_true': opinions_true, 'word_indices_of_aspect_terms': original_line_data['aspect_term']})

        return result