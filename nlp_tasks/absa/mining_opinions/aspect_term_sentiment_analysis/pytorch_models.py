# -*- coding: utf-8 -*-


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
from allennlp.modules import attention
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit
from allennlp.nn import util as allennlp_util
import dgl
from dgl import function as dgl_fn
from dgl import DGLGraph

from nlp_tasks.utils import attention_visualizer
from nlp_tasks.absa.mining_opinions.sequence_labeling.pytorch_models import Estimator


class SpanBasedModel(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.polarites = polarities
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        num_layers = self.configuration['layer_number_of_lstm']
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)
        sentiment_fc_input_size = word_embedding_dim
        self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                          nn.ReLU(),
                                          nn.Linear(sentiment_fc_input_size, self.polarity_num))
        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)

        sentiment_outputs = []
        for i, element in enumerate(sample):
            sentiment_output_of_one_sample = []
            word_indices_of_aspect_terms = element['word_indices_of_aspect_terms']
            for j in range(self.configuration['max_aspect_term_num']):
                if j < len(word_indices_of_aspect_terms):
                    word_indices_of_aspect_term = word_indices_of_aspect_terms[j]
                    start_index = word_indices_of_aspect_term[0]
                    end_index = word_indices_of_aspect_term[1]
                    word_representations = lstm_result[i][start_index: end_index]
                    aspect_term_word_num = end_index - start_index
                    if aspect_term_word_num > 1:
                        aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                        aspect_term_representation = aspect_term_representation.unsqueeze(0)
                    else:
                        aspect_term_representation = word_representations
                else:
                    aspect_term_representation = lstm_result[i][-1].unsqueeze(0)
                sentiment_output_of_one_sample.append(self.sentiment_fc(aspect_term_representation))
            sentiment_output_of_one_sample_cat = torch.cat(sentiment_output_of_one_sample, dim=0)
            sentiment_outputs.append(sentiment_output_of_one_sample_cat.unsqueeze(0))
        sentiment_outputs_cat = torch.cat(sentiment_outputs, dim=0)

        output = {}
        if label is not None:
            final_sentiment_outputs = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.configuration['max_aspect_term_num']):
                final_sentiment_outputs.append(sentiment_outputs_cat[:, i])
                polarity_labels.append(label[:, i])
                polarity_masks.append(polarity_mask[:, i])

            output['final_sentiment_outputs'] = final_sentiment_outputs
            output['polarity_labels'] = polarity_labels
            output['polarity_masks'] = polarity_masks

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)

            loss = self.sentiment_loss(sentiment_logit, sentiment_label.long())
            if torch.isnan(loss):
                print()

            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            output['logit'] = sentiment_logit
            output['label'] = sentiment_label
            output['mask'] = sentiment_mask
            output['loss'] = loss

        # 可视化，用模型对每个词进行情感预测，看看预测结果
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words']
                word_representations = []
                for j in range(len(words)):
                    word_representations.append(lstm_result[i][j])
                prediction_of_words = [self.sentiment_fc(word_representation) for word_representation in word_representations]
                prediction_of_words = [torch.softmax(prediction, dim=-1).detach().numpy() for prediction in prediction_of_words]

                titles_sentiment = []
                title_array = []
                for j, aspect_term in enumerate(sample[i]['aspect_terms']):
                    term = aspect_term.term
                    polarity = aspect_term.polarity
                    prediction_of_term = torch.softmax(sentiment_outputs_cat[i][j], dim=-1).detach().numpy()
                    title_array.append(term)
                    title_array.append(polarity)
                    title_array.append(str(prediction_of_term))
                title_array.append(str(self.polarites))
                titles_sentiment.append('-'.join(title_array))
                titles_sentiment.extend([''] * 2)
                labels_sentiment = []
                visual_sentiment = []
                for k in range(self.polarity_num):
                    labels_sentiment.append(self.polarites[k])
                    visual_sentiment.append([prediction[k] for prediction in prediction_of_words])
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_sentiment,
                                                                       labels_sentiment,
                                                                       titles_sentiment)


        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset)
        }
        return metrics


class SpanBasedBertModel(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.polarites = polarities
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        sentiment_fc_input_size = lstm_input_size
        self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                          nn.ReLU(),
                                          nn.Linear(sentiment_fc_input_size, self.polarity_num))
        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        # word_embeddings = self.word_embedder(tokens)

        token_type_ids = tokens['tokens-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = tokens['tokens-offsets']
        word_embeddings = self.word_embedder(tokens, token_type_ids=token_type_ids, offsets=offsets)

        word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        lstm_result = word_embeddings

        sentiment_outputs = []
        for i, element in enumerate(sample):
            sentiment_output_of_one_sample = []
            word_indices_of_aspect_terms = element['word_indices_of_aspect_terms']
            for j in range(self.configuration['max_aspect_term_num']):
                if j < len(word_indices_of_aspect_terms):
                    word_indices_of_aspect_term = word_indices_of_aspect_terms[j]
                    start_index = word_indices_of_aspect_term[0]
                    end_index = word_indices_of_aspect_term[1]
                    word_representations = lstm_result[i][start_index: end_index]
                    aspect_term_word_num = end_index - start_index
                    if aspect_term_word_num > 1:
                        aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                        aspect_term_representation = aspect_term_representation.unsqueeze(0)
                    else:
                        aspect_term_representation = word_representations
                else:
                    aspect_term_representation = lstm_result[i][-1].unsqueeze(0)
                sentiment_output_of_one_sample.append(self.sentiment_fc(aspect_term_representation))
            sentiment_output_of_one_sample_cat = torch.cat(sentiment_output_of_one_sample, dim=0)
            sentiment_outputs.append(sentiment_output_of_one_sample_cat.unsqueeze(0))
        sentiment_outputs_cat = torch.cat(sentiment_outputs, dim=0)

        output = {}
        if label is not None:
            final_sentiment_outputs = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.configuration['max_aspect_term_num']):
                final_sentiment_outputs.append(sentiment_outputs_cat[:, i])
                polarity_labels.append(label[:, i])
                polarity_masks.append(polarity_mask[:, i])

            output['final_sentiment_outputs'] = final_sentiment_outputs
            output['polarity_labels'] = polarity_labels
            output['polarity_masks'] = polarity_masks

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)

            loss = self.sentiment_loss(sentiment_logit, sentiment_label.long())
            if torch.isnan(loss):
                for e in sample:
                    print('text: %s' % e['text'])
                    print('words: %s' % e['words'])

            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            output['logit'] = sentiment_logit
            output['label'] = sentiment_label
            output['mask'] = sentiment_mask
            output['loss'] = loss

        # 可视化，用模型对每个词进行情感预测，看看预测结果
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words']
                word_representations = []
                for j in range(len(words)):
                    word_representations.append(lstm_result[i][j])
                prediction_of_words = [self.sentiment_fc(word_representation) for word_representation in word_representations]
                prediction_of_words = [torch.softmax(prediction, dim=-1).detach().numpy() for prediction in prediction_of_words]

                titles_sentiment = []
                title_array = []
                for j, aspect_term in enumerate(sample[i]['aspect_terms']):
                    term = aspect_term.term
                    polarity = aspect_term.polarity
                    prediction_of_term = torch.softmax(sentiment_outputs_cat[i][j], dim=-1).detach().numpy()
                    title_array.append(term)
                    title_array.append(polarity)
                    title_array.append(str(prediction_of_term))
                title_array.append(str(self.polarites))
                titles_sentiment.append('-'.join(title_array))
                titles_sentiment.extend([''] * 2)
                labels_sentiment = []
                visual_sentiment = []
                for k in range(self.polarity_num):
                    labels_sentiment.append(self.polarites[k])
                    visual_sentiment.append([prediction[k] for prediction in prediction_of_words])
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_sentiment,
                                                                       labels_sentiment,
                                                                       titles_sentiment)


        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset)
        }
        return metrics


class AtsaBERT(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.polarites = polarities
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        if self.configuration['mean_or_cat_of_term_and_cls'] == 'cat':
            sentiment_fc_input_size = lstm_input_size * 2
        else:
            # mean
            sentiment_fc_input_size = lstm_input_size
        self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, word_embedding_dim),
                                          nn.ReLU(),
                                          nn.Linear(word_embedding_dim, self.polarity_num))
        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        # word_embeddings = self.word_embedder(tokens)

        token_type_ids = tokens['tokens-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = tokens['tokens-offsets']
        word_embeddings = self.word_embedder(tokens, token_type_ids=token_type_ids, offsets=offsets)

        word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        lstm_result = word_embeddings

        sentiment_outputs = []
        for i, element in enumerate(sample):
            sentiment_output_of_one_sample = [] # [(1,3), ...]
            word_indices_of_aspect_terms = element['word_indices_of_aspect_terms']
            cls_representation = lstm_result[i][0].unsqueeze(dim=0)
            for j in range(self.configuration['max_aspect_term_num']):
                if j < len(word_indices_of_aspect_terms):
                    word_indices_of_aspect_term = word_indices_of_aspect_terms[j]
                    start_index = word_indices_of_aspect_term[0]
                    end_index = word_indices_of_aspect_term[1]
                    word_representations = lstm_result[i][start_index: end_index]
                    aspect_term_word_num = end_index - start_index
                    if aspect_term_word_num > 1:
                        aspect_term_representation= torch.mean(word_representations, dim=0)
                        # aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                        aspect_term_representation = aspect_term_representation.unsqueeze(0)
                    else:
                        aspect_term_representation = word_representations
                else:
                    aspect_term_representation = lstm_result[i][-1].unsqueeze(0)
                if self.configuration['term'] and not self.configuration['cls']:
                    aspect_term_representation_final = aspect_term_representation
                elif not self.configuration['term'] and self.configuration['cls']:
                    aspect_term_representation_final = cls_representation
                else:
                    if self.configuration['mean_or_cat_of_term_and_cls'] == 'cat':
                        aspect_term_representation_final = torch.cat([cls_representation, aspect_term_representation], dim=1) # (1, 1536)
                    else:
                        cls_term_representation_cat = torch.cat([cls_representation, aspect_term_representation], dim=0)
                        aspect_term_representation_final = torch.mean(cls_term_representation_cat, dim=0).unsqueeze(dim=0) # (1,768)
                sentiment_output_of_one_sample.append(self.sentiment_fc(aspect_term_representation_final))
            sentiment_output_of_one_sample_cat = torch.cat(sentiment_output_of_one_sample, dim=0) # (11,3)
            sentiment_outputs.append(sentiment_output_of_one_sample_cat.unsqueeze(0))
        sentiment_outputs_cat = torch.cat(sentiment_outputs, dim=0) # [(32,11,3)]

        output = {}
        if label is not None:
            final_sentiment_outputs = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.configuration['max_aspect_term_num']):
                final_sentiment_outputs.append(sentiment_outputs_cat[:, i])
                polarity_labels.append(label[:, i])
                polarity_masks.append(polarity_mask[:, i])

            output['final_sentiment_outputs'] = final_sentiment_outputs
            output['polarity_labels'] = polarity_labels
            output['polarity_masks'] = polarity_masks

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)

            loss = self.sentiment_loss(sentiment_logit, sentiment_label.long())
            if torch.isnan(loss):
                for e in sample:
                    print('text: %s' % e['text'])
                    print('words: %s' % e['words'])

            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            output['logit'] = sentiment_logit
            output['label'] = sentiment_label
            output['mask'] = sentiment_mask
            output['loss'] = loss

        # 可视化，用模型对每个词进行情感预测，看看预测结果
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words']
                word_representations = []
                for j in range(len(words)):
                    word_representations.append(lstm_result[i][j])
                prediction_of_words = [self.sentiment_fc(word_representation) for word_representation in word_representations]
                prediction_of_words = [torch.softmax(prediction, dim=-1).detach().numpy() for prediction in prediction_of_words]

                titles_sentiment = []
                title_array = []
                for j, aspect_term in enumerate(sample[i]['aspect_terms']):
                    term = aspect_term.term
                    polarity = aspect_term.polarity
                    prediction_of_term = torch.softmax(sentiment_outputs_cat[i][j], dim=-1).detach().numpy()
                    title_array.append(term)
                    title_array.append(polarity)
                    title_array.append(str(prediction_of_term))
                title_array.append(str(self.polarites))
                titles_sentiment.append('-'.join(title_array))
                titles_sentiment.extend([''] * 2)
                labels_sentiment = []
                visual_sentiment = []
                for k in range(self.polarity_num):
                    labels_sentiment.append(self.polarites[k])
                    visual_sentiment.append([prediction[k] for prediction in prediction_of_words])
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_sentiment,
                                                                       labels_sentiment,
                                                                       titles_sentiment)


        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset)
        }
        return metrics


class GatEdge:
    """
    story information of a edge in gat for visualization
    """

    def __init__(self, src_ids: List[int], dst_id: int, alphas: List[float]):
        self.dst_id = dst_id
        self.src_ids = src_ids
        self.alphas = alphas

    def add(self, other_edge: 'GatEdge'):
        """

        :param other_edge:  同样的边
        :return:
        """
        if self.src_ids != other_edge.src_ids or self.dst_id != other_edge.dst_id:
            print('add error')
            return
        for i in range(len(self.alphas)):
            self.alphas[i] += other_edge.alphas[i]

    def divide(self, number: int):
        self.alphas = [alpha / number for alpha in self.alphas]

    def __str__(self):
        return '%s-%s-%s' % (str(self.dst_id), str(self.src_ids), str(self.alphas))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, opt: dict={}):
        super().__init__()
        self.opt = opt
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges: dgl.EdgeBatch):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges: dgl.EdgeBatch):
        # message UDF for equation (3) & (4)
        result = {'z': edges.src['z'], 'node_ids': edges.src['node_ids'], 'e': edges.data['e']}
        return result

    def reduce_func(self, nodes: dgl.NodeBatch):
        # for visualization
        src_node_ids = nodes.mailbox['node_ids']
        dst_node_ids = nodes.data['node_ids']

        # reduce UDF for equation (3) & (4)
        # equation (3)

        e = nodes.mailbox['e']
        alpha = F.softmax(e, dim=1)
        # equation (4)
        z = nodes.mailbox['z']
        h = torch.sum(alpha * z, dim=1)

        self.for_visualization.append({'src_node_ids': src_node_ids, 'dst_node_ids': dst_node_ids,
                                       'alpha': alpha})

        return {'h': h}

    def forward(self, h, g: List[DGLGraph]):
        self.for_visualization = []

        batched_graph = dgl.batch(g)
        feature = h.view([-1, h.size()[-1]])

        # equation (1)
        z = self.fc(feature)
        batched_graph.ndata['z'] = z

        # 加入node id，用于attention可视化
        node_ids = batched_graph.nodes()
        batched_graph.ndata['node_ids'] = node_ids

        # equation (2)
        batched_graph.apply_edges(self.edge_attention)
        # equation (3) & (4)
        batched_graph.update_all(self.message_func, self.reduce_func)

        ug = dgl.unbatch(batched_graph)
        # output = [torch.unsqueeze(g.ndata.pop('h'), 0).to(self.opt['device']) for g in ug]
        # for visualization in local machine
        output = [torch.unsqueeze(g.ndata.pop('h'), 0) for g in ug]
        output = torch.cat(output, 0)

        # 对for_visualization按照sample进行拆分
        sample_num = len(g)
        node_num_per_sample = h.shape[1]
        edges_of_samples = [[] for _ in range(sample_num)]
        if 'gat_visualization' not in self.opt or self.opt['gat_visualization']:
            for edges in self.for_visualization:
                edge_num = edges['dst_node_ids'].shape[0]
                for i in range(edge_num):
                    src_ids = edges['src_node_ids'][i].cpu().numpy().tolist()
                    src_ids_real = [e % node_num_per_sample for e in src_ids]
                    dst_id = edges['dst_node_ids'][i].cpu().numpy().tolist()
                    dst_id_real = dst_id % node_num_per_sample
                    sample_index = dst_id // node_num_per_sample
                    alphas = edges['alpha'][i].squeeze(dim=-1).cpu().numpy().tolist()
                    edge = GatEdge(src_ids_real, dst_id_real, alphas)
                    edges_of_samples[sample_index].append(edge)

        return output, edges_of_samples


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', opt: dict={}):
        super().__init__()
        self.opt = opt
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, opt))
        self.merge = merge

    def forward(self, h, g: List[DGLGraph]):
        head_outs = [attn_head(h, g) for attn_head in self.heads]
        head_outs_feature = [e[0] for e in head_outs]
        head_outs_attention = [e[1] for e in head_outs]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs_feature, dim=2), head_outs_attention
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs_feature)), head_outs_attention


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, opt: dict={}):
        super().__init__()
        self.opt = opt
        self.layer1 = MultiHeadGATLayer(in_dim, int(out_dim / num_heads), num_heads, opt=opt)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h, g: List[DGLGraph]):
        h, attention = self.layer1(h, g)
        # h = F.elu(h)
        # h = self.layer2(h, g)
        return h, attention


class SyntaxAwareSpanBasedBertModel(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.polarites = polarities
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        sentiment_fc_input_size = lstm_input_size
        self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                          nn.ReLU(),
                                          nn.Linear(sentiment_fc_input_size, self.polarity_num))
        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)

        self.gnn_for_sentiment = GAT(word_embedding_dim, word_embedding_dim, word_embedding_dim, 4, self.configuration)

    def pad_dgl_graph(self, graphs, max_node_num):
        graphs_padded = []
        for graph in graphs:
            graph_padded = copy.deepcopy(graph)
            node_num = graph.number_of_nodes()
            graph_padded.add_nodes(max_node_num - node_num)
            graphs_padded.append(graph_padded)
        return graphs_padded

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        # word_embeddings = self.word_embedder(tokens)

        token_type_ids = tokens['tokens-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = tokens['tokens-offsets']
        word_embeddings = self.word_embedder(tokens, token_type_ids=token_type_ids, offsets=offsets)

        # 加图
        max_len = mask.size()[1]
        graphs = [e['graph'] for e in sample]
        graphs_padded = self.pad_dgl_graph(graphs, max_len)
        word_embeddings, graph_attention_weights = self.gnn_for_sentiment(word_embeddings, graphs_padded)

        word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        lstm_result = word_embeddings

        sentiment_outputs = []
        for i, element in enumerate(sample):
            sentiment_output_of_one_sample = []
            word_indices_of_aspect_terms = element['word_indices_of_aspect_terms']
            for j in range(self.configuration['max_aspect_term_num']):
                if j < len(word_indices_of_aspect_terms):
                    word_indices_of_aspect_term = word_indices_of_aspect_terms[j]
                    start_index = word_indices_of_aspect_term[0]
                    end_index = word_indices_of_aspect_term[1]
                    word_representations = lstm_result[i][start_index: end_index]
                    aspect_term_word_num = end_index - start_index
                    if aspect_term_word_num > 1:
                        aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                        aspect_term_representation = aspect_term_representation.unsqueeze(0)
                    else:
                        aspect_term_representation = word_representations
                else:
                    aspect_term_representation = lstm_result[i][-1].unsqueeze(0)
                sentiment_output_of_one_sample.append(self.sentiment_fc(aspect_term_representation))
            sentiment_output_of_one_sample_cat = torch.cat(sentiment_output_of_one_sample, dim=0)
            sentiment_outputs.append(sentiment_output_of_one_sample_cat.unsqueeze(0))
        sentiment_outputs_cat = torch.cat(sentiment_outputs, dim=0)

        output = {}
        if label is not None:
            final_sentiment_outputs = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.configuration['max_aspect_term_num']):
                final_sentiment_outputs.append(sentiment_outputs_cat[:, i])
                polarity_labels.append(label[:, i])
                polarity_masks.append(polarity_mask[:, i])

            output['final_sentiment_outputs'] = final_sentiment_outputs
            output['polarity_labels'] = polarity_labels
            output['polarity_masks'] = polarity_masks

            # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)

            loss = self.sentiment_loss(sentiment_logit, sentiment_label.long())
            if torch.isnan(loss):
                for e in sample:
                    print('text: %s' % e['text'])
                    print('words: %s' % e['words'])

            self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            output['logit'] = sentiment_logit
            output['label'] = sentiment_label
            output['mask'] = sentiment_mask
            output['loss'] = loss

        # 可视化，用模型对每个词进行情感预测，看看预测结果
        if self.configuration['visualize_attention']:
            for i in range(len(sample)):
                words = sample[i]['words']
                word_representations = []
                for j in range(len(words)):
                    word_representations.append(lstm_result[i][j])
                prediction_of_words = [self.sentiment_fc(word_representation) for word_representation in word_representations]
                prediction_of_words = [torch.softmax(prediction, dim=-1).detach().numpy() for prediction in prediction_of_words]

                titles_sentiment = []
                title_array = []
                for j, aspect_term in enumerate(sample[i]['aspect_terms']):
                    term = aspect_term.term
                    polarity = aspect_term.polarity
                    prediction_of_term = torch.softmax(sentiment_outputs_cat[i][j], dim=-1).detach().numpy()
                    title_array.append(term)
                    title_array.append(polarity)
                    title_array.append(str(prediction_of_term))
                title_array.append(str(self.polarites))
                titles_sentiment.append('-'.join(title_array))
                titles_sentiment.extend([''] * 2)
                labels_sentiment = []
                visual_sentiment = []
                for k in range(self.polarity_num):
                    labels_sentiment.append(self.polarites[k])
                    visual_sentiment.append([prediction[k] for prediction in prediction_of_words])
                attention_visualizer.plot_multi_attentions_of_sentence(words, visual_sentiment,
                                                                       labels_sentiment,
                                                                       titles_sentiment)


        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset)
        }
        return metrics


class SpanBasedModelV2(Model):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 polarities: list, vocab: Vocabulary, configuration: dict):
        super().__init__(vocab)
        self.configuration = configuration
        self.word_embedder = word_embedder
        self.position_embedder = position_embedder
        self.polarites = polarities
        self.polarity_num = len(polarities)
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._accuracy = metrics.CategoricalAccuracy()

        word_embedding_dim = word_embedder.get_output_dim()
        lstm_input_size = word_embedding_dim
        num_layers = 3
        self.lstm = torch.nn.LSTM(lstm_input_size, int(word_embedding_dim / 2), batch_first=True,
                                  bidirectional=True, num_layers=num_layers, dropout=0.5)
        sentiment_fc_input_size = word_embedding_dim
        self.sentiment_fc = nn.Sequential(nn.Linear(sentiment_fc_input_size, sentiment_fc_input_size),
                                          nn.ReLU(),
                                          nn.Linear(sentiment_fc_input_size, self.polarity_num))
        self.dropout_after_embedding_layer = nn.Dropout(0.5)
        self.dropout_after_lstm_layer = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor, position: torch.Tensor,
                polarity_mask: torch.Tensor, sample: list) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding_layer(word_embeddings)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm_layer(lstm_result)

        sentiment_logit = []
        for i, element in enumerate(sample):
            word_indices_of_aspect_terms = element['word_indices_of_aspect_terms']
            for j in range(len(word_indices_of_aspect_terms)):
                word_indices_of_aspect_term = word_indices_of_aspect_terms[j]
                start_index = word_indices_of_aspect_term[0]
                end_index = word_indices_of_aspect_term[1]
                word_representations = lstm_result[i][start_index: end_index]
                aspect_term_word_num = end_index - start_index
                if aspect_term_word_num > 1:
                    aspect_term_representation = torch.sum(word_representations, dim=0) / len(word_representations)
                    aspect_term_representation = aspect_term_representation.unsqueeze(0)
                else:
                    aspect_term_representation = word_representations
                sentiment_logit.append(self.sentiment_fc(aspect_term_representation))

        output = {}
        if label is not None:
            # sentiment accuracy
            sentiment_label = []
            for i, element in enumerate(sample):
                word_indices_of_aspect_terms = element['word_indices_of_aspect_terms']
                for j in range(len(word_indices_of_aspect_terms)):
                    sentiment_label.append(label[i][j])

            loss = self.sentiment_loss(sentiment_logit, sentiment_label.long())
            if torch.isnan(loss):
                print()

            self._accuracy(sentiment_logit, sentiment_label)

            output['logit'] = sentiment_logit
            output['label'] = sentiment_label
            output['mask'] = None
            output['loss'] = loss
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            'accuracy': self._accuracy.get_metric(reset)
        }
        return metrics


class SpanBasedModelEstimator(Estimator):
    def __init__(self, model: Model, iterator: DataIterator, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self.cuda_device = cuda_device
        self.configuration = configuration

    def estimate(self, ds: Iterable[Instance]) -> dict:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            sentiment_logits = []
            sentiment_labels = []
            sentiment_masks = []
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output = self.model(**batch)
                sentiment_logit = output['logit']
                sentiment_label = output['label']
                sentiment_mask = output['mask']

                sentiment_logits.append(sentiment_logit)
                sentiment_labels.append(sentiment_label)
                sentiment_masks.append(sentiment_mask)
            sentiment_logit_total = torch.cat(sentiment_logits, dim=0)
            sentiment_label_total = torch.cat(sentiment_labels, dim=0)
            sentiment_mask_total = torch.cat(sentiment_masks, dim=0)
            self._accuracy(sentiment_logit_total, sentiment_label_total, sentiment_mask_total)
        return {'sentiment_acc': self._accuracy.get_metric(reset=True)}


class predictor:

    def predict(self, ds: Iterable[Instance]) -> dict:
        raise NotImplementedError('predict')


class SpanBasedModelPredictor(predictor):
    def __init__(self, model: Model, iterator: DataIterator, polarities: list,
                 cuda_device: int = -1, configuration: dict=None) -> None:
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.polarities = polarities
        self._accuracy = metrics.CategoricalAccuracy()
        self.cuda_device = cuda_device
        self.configuration = configuration

    def predict(self, ds: Iterable[Instance]) -> dict:
        self.model.eval()
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            result = []
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output = self.model(**batch)
                final_sentiment_outputs = output['final_sentiment_outputs']
                polarity_labels = output['polarity_labels']
                polarity_masks = output['polarity_masks']
                sample = batch['sample']
                for i, one_sample in enumerate(sample):
                    final_sentiment_outputs_for_one_sample = [torch.softmax(e[i], dim=-1) for e in final_sentiment_outputs]
                    sentiment_outputs_for_aspect_terms = []
                    for j, aspect_term in enumerate(one_sample['aspect_terms']):
                        aspect_term_sentiment_output = final_sentiment_outputs_for_one_sample[j]
                        aspect_term_label_index_predict = aspect_term_sentiment_output.argmax(dim=-1)
                        aspect_term_label_predict = self.polarities[aspect_term_label_index_predict]
                        sentiment_outputs_for_aspect_terms.append((aspect_term_sentiment_output.cpu().numpy().tolist(),
                                                                   aspect_term_label_predict))
                    one_sample['sentiment_outputs_for_aspect_terms'] = sentiment_outputs_for_aspect_terms
                    result.append(one_sample)
        return result