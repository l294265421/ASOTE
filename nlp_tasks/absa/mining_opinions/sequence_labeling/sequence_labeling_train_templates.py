import numpy as np
import os
import logging
import sys
import pickle
from typing import List
import json

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from allennlp.data.token_indexers import WordpieceIndexer
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import BasicIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
import torch.optim as optim
from torch.optim import adagrad
# from allennlp.training.trainer import Trainer
from nlp_tasks.absa.mining_opinions.sequence_labeling.my_allennlp_trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.predictors import text_classifier
from allennlp.data.token_indexers import SingleIdTokenIndexer
import spacy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm
from allennlp.nn import util as nn_util
from allennlp.data.dataset_readers import DatasetReader
from allennlp.modules.token_embedders.bert_token_embedder import BertModel, PretrainedBertModel
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder

from nlp_tasks.absa.entities import ModelTrainTemplate
from nlp_tasks.absa.mining_opinions.sequence_labeling import sequence_labeling_data_reader
from nlp_tasks.absa.mining_opinions.sequence_labeling import pytorch_models
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from nlp_tasks.utils import file_utils
from nlp_tasks.common import common_path
from nlp_tasks.absa.aspect_category_detection_and_sentiment_classification import allennlp_callback
from nlp_tasks.common import common_path
from nlp_tasks.utils import file_utils
from nlp_tasks.absa.mining_opinions.data_adapter import data_object
from nlp_tasks.utils import word_processor
from nlp_tasks.utils import tokenizers
from nlp_tasks.utils import tokenizer_wrappers

task_dir = common_path.get_task_data_dir('absa')


class ModelTrainTemplate:
    """
    1.
    2.
    3.
    4.
    5.
    """

    def __init__(self, configuration: dict):
        self.configuration = configuration
        # 、、；，
        #
        if 'data_type' not in self.configuration:
            self.configuration['data_type'] = 'common'
        self.base_data_dir = task_dir + ('{task_name}/{current_dataset}/{data_type}/'\
            .format_map(self.configuration))
        if not os.path.exists(self.base_data_dir):
            os.makedirs(self.base_data_dir)
        self.embedding_matrix_file_path = self.base_data_dir + 'embedding_matrix'
        self.keras_tokenizer_file_path = self.base_data_dir + 'keras_tokenizer'

        self.base_model_dir = self.base_data_dir + ('{model_name_complete}/{timestamp}/'.format_map(self.configuration))
        self.model_dir = self.base_model_dir + 'models/'
        if self.configuration['train'] and os.path.exists(self.model_dir):
            file_utils.rm_r(self.model_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_model_filepath = self.model_dir + self.configuration['model_name'] + '.hdf5'
        self.model_meta_data_filepath = self.model_dir + self.configuration['model_name'] + '.model_meta_data'
        self.model_meta_data = {}
        self.model = None
        if not self.configuration['train']:
            self._load_model_meta_data()
            self._load_model()

        self.model_log_dir = self.base_model_dir + 'model-log/'
        if not os.path.exists(self.model_log_dir):
            os.makedirs(self.model_log_dir)

        # log
        logger_name = 'performance'
        log_filepath = self.base_model_dir + ('%s.log' % logger_name)
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        logging_level = logging.INFO
        self.logger.setLevel(logging_level)

        formatter = logging.Formatter('%(filename)s-%(lineno)d-%(asctime)s-%(message)s')

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging_level)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        file = logging.FileHandler(log_filepath)
        file.setLevel(logging_level)
        file.setFormatter(formatter)
        self.logger.addHandler(file)

        self.dataset = self._get_dataset()

    def _get_dataset(self):
        return data_object.get_dataset_class_by_name(self.configuration['current_dataset'])(self.configuration)

    def _load_word_vec(self, path, word2idx=None):
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            if word2idx is None or tokens[0] in word2idx.keys():
                try:
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
                except:
                    continue
        return word_vec

    def _build_embedding_matrix(self, embedding_filepath, word2idx, embed_dim):
        if os.path.exists(self.embedding_matrix_file_path):
            print('loading embedding_matrix:', self.embedding_matrix_file_path)
            embedding_matrix = pickle.load(open(self.embedding_matrix_file_path, 'rb'))
        else:
            print('loading word vectors ...')
            embedding_matrix = np.zeros((len(word2idx) + 1, embed_dim))  # idx 0 and 1 are all-zeros
            embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
            word_vec = self._load_word_vec(embedding_filepath, word2idx=word2idx)
            print('building embedding_matrix:', self.embedding_matrix_file_path)
            for word, i in word2idx.items():
                vec = word_vec.get(word)
                if vec is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = vec
            pickle.dump(embedding_matrix, open(self.embedding_matrix_file_path, 'wb'))
        return embedding_matrix

    def _load_data(self):
        pass

    def _save_model(self):
        pass

    def _save_model_meta_data(self):
        with open(self.model_meta_data_filepath, mode='wb') as meta_data_file:
            pickle.dump(self.model_meta_data, meta_data_file)

    def _load_model_meta_data(self):
        with open(self.model_meta_data_filepath, mode='rb') as meta_data_file:
            self.model_meta_data = pickle.load(meta_data_file)

    def _save_object(self, filepath, data):
        with open(filepath, mode='wb') as data_file:
            pickle.dump(data, data_file)

    def _load_object(self, filepath):
        with open(filepath, mode='rb') as data_file:
            data = pickle.load(data_file)
            return data

    def _load_model(self, return_model_meta_data=False):
        pass

    def _get_word_segmenter(self):
        word_processor1 = word_processor.LowerProcessor()
        word_segmenter = tokenizers.SpacyTokenizer(word_processor=word_processor1)
        return word_segmenter

    def _get_keras_tokenizer(self, texts):
        if not os.path.exists(self.keras_tokenizer_file_path):
            word_segmenter = self._get_word_segmenter()
            keras_tokenizer = tokenizer_wrappers.TokenizerWithCustomWordSegmenter(word_segmenter, oov_token='oov')
            keras_tokenizer.fit_on_texts(texts)
            self._save_object(self.keras_tokenizer_file_path, keras_tokenizer)
        else:
            keras_tokenizer = self._load_object(self.keras_tokenizer_file_path)
        return keras_tokenizer

    def _find_model_function(self):
        """
        ，
        :return:
        """
        pass

    def _transform_data_for_model(self):
        """

        :return:
        """
        pass

    def _transform_label_for_model(self):
        pass

    def _get_word_index(self, *args):
        pass

    def _inner_train(self):
        pass

    def train(self):
        self._inner_train()
        self._save_model_meta_data()
        self._save_model()
        self._load_model()

    def evaluate(self):
        pass

    def observe(self):
        pass


class SequenceLabeling(ModelTrainTemplate):
    """
    """

    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_reader: DatasetReader = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self._load_data()
        if self.configuration['debug']:
            self.train_data = self.train_data[: 128]
            self.dev_data = self.dev_data[: 128]
            self.test_data = self.test_data[: 128]

        self.vocab = None
        self._build_vocab()

        self.iterator = None
        self.val_iterator = None
        self._build_iterator()

    def _get_data_reader(self):
        raise NotImplementedError()

    def _load_data(self):
        reader = self._get_data_reader()
        self.data_reader = reader

        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, = super()._load_object(data_filepath)
        else:
            train_dev_test_data = self.dataset.get_data_type_and_data_dict()

            train_dev_test_data_new = {}
            for data_type, data in train_dev_test_data.items():
                data_new = []
                for sample in data:
                    sample_new = {
                        'words': sample.words,
                        'target_tags': sample.target_tags,
                        'opinion_words_tags': sample.opinion_words_tags,
                        'polarity': sample.polarity,
                        'metadata': sample.metadata
                    }
                    data_new.append(sample_new)
                train_dev_test_data_new[data_type] = data_new

            self.train_data = reader.read(train_dev_test_data_new['train'])
            self.dev_data = reader.read(train_dev_test_data_new['dev'])
            self.test_data = reader.read(train_dev_test_data_new['test'])
            data = [self.train_data, self.dev_data, self.test_data]
            super()._save_object(data_filepath, data)

    def _build_vocab(self):
        if self.configuration['train']:
            vocab_file_path = self.base_data_dir + 'vocab'
            if os.path.exists(vocab_file_path):
                self.vocab = super()._load_object(vocab_file_path)
            else:
                data = self.train_data + self.dev_data + self.test_data
                self.vocab = Vocabulary.from_instances(data, max_vocab_size=sys.maxsize)
                super()._save_object(vocab_file_path, self.vocab)
            self.model_meta_data['vocab'] = self.vocab
        else:
            self.vocab = self.model_meta_data['vocab']

    def _build_iterator(self):
        self.iterator = BucketIterator(batch_size=self.configuration['batch_size'],
                                       sorting_keys=[("tokens", "num_tokens")],
                                       )
        self.iterator.index_with(self.vocab)
        self.val_iterator = BasicIterator(batch_size=self.configuration['batch_size'])
        self.val_iterator.index_with(self.vocab)

    def _print_args(self, model):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in self.configuration.keys():
            self.logger.info('>>> {0}: {1}'.format(arg, self.configuration[arg]))

    def _find_model_function_pure(self):
        raise NotImplementedError()

    def _get_position_embeddings_dim(self):
        return 300

    def _is_train_token_embeddings(self):
        return False

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        embedding_matrix = embedding_matrix.to(self.configuration['device'])
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=self._is_train_token_embeddings(), weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
        )
        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params, lr=0.001, weight_decay=0.00001)

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.SequenceLabelingModelEstimator(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)
        return estimator

    def _get_estimate_callback(self, model):
        result = []
        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        estimator = self._get_estimator(model)
        estimate_callback = allennlp_callback.EstimateCallback(data_type_and_data, estimator, self.logger)
        result.append(estimate_callback)
        return result

    def _get_loss_weight_callback(self):
        result = []
        set_loss_weight_callback = allennlp_callback.SetLossWeightCallback(self.model, self.logger,
                                                                           acd_warmup_epoch_num=self._get_acd_warmup_epoch_num())
        result.append(set_loss_weight_callback)
        return result

    def _get_fixed_loss_weight_callback(self, model, loss_weights: dict):
        result = []
        fixed_loss_weight_callback = allennlp_callback.LossWeightCallback(model, self.logger, loss_weights)
        result.append(fixed_loss_weight_callback)
        return result

    def _get_bert_word_embedder(self):
        return None

    def _inner_train(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1

        self.model = self._find_model_function()
        estimator = self._get_estimator(self.model)
        callbacks = self._get_estimate_callback(self.model)
        validation_metric = '+f1'
        if 'validation_metric' in self.configuration:
            validation_metric = self.configuration['validation_metric']
        self.logger.info('validation_metric: %s' % validation_metric)
        optimizer = self._get_optimizer(self.model)
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            cuda_device=gpu_id,
            num_epochs=self.configuration['epochs'],
            validation_metric=validation_metric,
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=self.configuration['patience'],
            callbacks=callbacks,
            num_serialized_models_to_keep=0,
            early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
            estimator=estimator,
            grad_clipping=5
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))

    def _save_model(self):
        torch.save(self.model, self.best_model_filepath)

    def _load_model(self):
        if torch.cuda.is_available():
            self.model = torch.load(self.best_model_filepath)
        else:
            self.model = torch.load(self.best_model_filepath, map_location=torch.device('cpu'))
        self.model.configuration = self.configuration

    def evaluate(self):
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data)
            self.logger.info('data_type: %s result: %s' % (data_type, result))

    def predict(self, texts: List[str]):
        instances = self.data_reader.read(texts)

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.SequenceLabelingModelPredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        for i, sample in enumerate(texts):
            sample['predicted_tags'] = result[i]
            print('----------------------------------------------------------')
            print('%s' % ','.join(['%s-%s' % (sample['words'][j], sample['target_tags'][j]) for j in range(len(sample['words']))]))
            print('%s' % ','.join(['%s-%s' % (sample['words'][j], result[i][j]) for j in range(len(sample['words']))]))
        return result

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

    def predict_test(self, output_filepath_prefix, only_error=False):
        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        for data_type, data in data_type_and_data.items():
            if data_type == 'test':
                output_filepath = output_filepath_prefix
            else:
                output_filepath = '%s.%s' % (output_filepath_prefix, data_type)
            instances = data

            USE_GPU = torch.cuda.is_available()
            if USE_GPU:
                gpu_id = self.configuration['gpu_id']
            else:
                gpu_id = -1
            predictor = pytorch_models.SequenceLabelingModelPredictor(self.model, self.val_iterator,
                                                                      cuda_device=gpu_id, configuration=self.configuration)

            result = predictor.predict(instances)
            output_lines = []
            for i in range(len(instances)):
                instance = instances[i]
                words = instance.fields['sample'].metadata['words']
                # text = instance.fields['sample'].metadata['metadata']['original_line'].split('####')[0]
                text = ' '.join(words)
                pred = result[i][: len(words)]
                term_with_texts = self.terms_from_tags(pred, words)
                try:
                    gold_term_with_text = self.terms_from_tags(instance.fields['sample'].metadata['all_target_tags'], words)
                except:
                    gold_term_with_text = []
                line = json.dumps({'text': text, 'pred': term_with_texts, 'true': gold_term_with_text}, ensure_ascii=False)
                if only_error:
                    if term_with_texts != gold_term_with_text:
                        output_lines.append(line)
                else:
                    output_lines.append(line)
            file_utils.write_lines(output_lines, output_filepath)


class SimpleSequenceLabeling(SequenceLabeling):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.SimpleSequenceLabelingDatasetReader(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.SimpleSequenceLabelingModel


class TCBiLSTMWithCrfTagger(SequenceLabeling):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.DatasetReaderForTCBiLSTM(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.TCBiLSTMWithCrfTagger


class ToweModel(SequenceLabeling):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def evaluate_v2(self):
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            # 'train': self.train_data,
            # 'dev': self.dev_data,
            # 'test': self.test_data
        }

        reader = self.data_reader

        train_dev_test_data = self.dataset.get_data_type_and_data_dict()
        for data_type in train_dev_test_data.keys():
            data_new = []
            for sample in train_dev_test_data[data_type]:
                sample_new = {
                    'words': sample.words,
                    'target_tags': sample.target_tags,
                    'opinion_words_tags': sample.opinion_words_tags,
                    'polarity': sample.polarity,
                    'metadata': sample.metadata
                }
                data_new.append(sample_new)

            instances = reader.read(data_new)
            data_type_and_data[data_type] = instances

        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data)
            self.logger.info('data_type: %s result: %s' % (data_type, result))

    def predict_test(self, output_filepath):
        instances = self.test_data

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.SequenceLabelingModelPredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        output_lines = []
        for i in range(len(instances)):
            instance = instances[i]
            # text = instance.fields['sample'].metadata['metadata']['original_line'].split('####')[0]
            text = instance.fields['sample'].metadata['metadata']['original_line_data']['sentence']
            words_real = text.split(' ')
            words = instance.fields['sample'].metadata['words']

            target_tags = instance.fields['sample'].metadata['target_tags']
            target = self.terms_from_tags(target_tags, words)[0]
            target_parts = target.split('-')
            target_parts[-2] = int(target_parts[-2])
            target_parts[-1] = int(target_parts[-1])

            pred = result[i][:len(words)]
            term_with_texts_temp = self.terms_from_tags(pred, words)
            term_with_texts = []
            if len(words_real) < len(words):
                for term_with_text in term_with_texts_temp:
                    term_with_text_parts = term_with_text.split('-')
                    if int(term_with_text_parts[-2]) < target_parts[-2]:
                        term_with_texts.append(term_with_text)
                    else:
                        term_with_text_parts[-2] = str(int(term_with_text_parts[-2]) - 2)
                        term_with_text_parts[-1] = str(int(term_with_text_parts[-1]) - 2)
                        term_with_texts.append('-'.join(term_with_text_parts))

                target_parts[-2] = str(target_parts[-2] - 1)
                target_parts[-1] = str(target_parts[-1] - 1)
            else:
                term_with_texts = term_with_texts_temp
                target_parts[-2] = str(target_parts[-2])
                target_parts[-1] = str(target_parts[-1])

            line = json.dumps({'text': text,
                               'pred': term_with_texts,
                               'aspect_terms': ['-'.join(target_parts)]},
                              ensure_ascii=False)
            output_lines.append(line)
        file_utils.write_lines(output_lines, output_filepath)

    def predict_test_v2(self, output_filepath):
        reader = self.data_reader

        train_dev_test_data = self.dataset.get_data_type_and_data_dict()

        data_new = []
        for sample in train_dev_test_data['test']:
            sample_new = {
                'words': sample.words,
                'target_tags': sample.target_tags,
                'opinion_words_tags': sample.opinion_words_tags,
                'polarity': sample.polarity,
                'metadata': sample.metadata
            }
            data_new.append(sample_new)

        instances = reader.read(data_new)

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.SequenceLabelingModelPredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        output_lines = []
        for i in range(len(instances)):
            instance = instances[i]
            # text = instance.fields['sample'].metadata['metadata']['original_line'].split('####')[0]
            text = instance.fields['sample'].metadata['metadata']['original_line_data']['sentence']
            words_real = text.split(' ')
            words = instance.fields['sample'].metadata['words']

            target_tags = instance.fields['sample'].metadata['target_tags']
            target = self.terms_from_tags(target_tags, words)[0]
            target_parts = target.split('-')
            target_parts[-2] = int(target_parts[-2])
            target_parts[-1] = int(target_parts[-1])

            pred = result[i][:len(words)]
            term_with_texts_temp = self.terms_from_tags(pred, words)
            term_with_texts = []
            if len(words_real) < len(words):
                for term_with_text in term_with_texts_temp:
                    term_with_text_parts = term_with_text.split('-')
                    if int(term_with_text_parts[-2]) < target_parts[-2]:
                        term_with_texts.append(term_with_text)
                    else:
                        term_with_text_parts[-2] = str(int(term_with_text_parts[-2]) - 2)
                        term_with_text_parts[-1] = str(int(term_with_text_parts[-1]) - 2)
                        term_with_texts.append('-'.join(term_with_text_parts))

                target_parts[-2] = str(target_parts[-2] - 1)
                target_parts[-1] = str(target_parts[-1] - 1)
            else:
                term_with_texts = term_with_texts_temp
                target_parts[-2] = str(target_parts[-2])
                target_parts[-1] = str(target_parts[-1])

            line = json.dumps({'text': text,
                               'pred': term_with_texts,
                               'aspect_terms': ['-'.join(target_parts)]},
                              ensure_ascii=False)
            output_lines.append(line)
        file_utils.write_lines(output_lines, output_filepath)


class TermBiLSTM(ToweModel):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.DatasetReaderForTermBiLSTM(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.TermBiLSTM

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.ToweEstimator(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)
        return estimator


class TermBiLSTMForMFGData(TermBiLSTM):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.DatasetReaderForTermBiLSTMForMFGData(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def predict_test(self, output_filepath):
        instances = self.test_data

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.SequenceLabelingModelPredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        output_lines = []
        for i in range(len(instances)):
            instance = instances[i]
            # text = instance.fields['sample'].metadata['metadata']['original_line'].split('####')[0]
            words_real = instance.fields['sample'].metadata['words_backup']
            text = ' '.join(words_real)
            words = instance.fields['sample'].metadata['words']

            target_tags = instance.fields['sample'].metadata['target_tags']
            target = self.terms_from_tags(target_tags, words)[0]
            target_parts = target.split('-')
            target_parts[-2] = int(target_parts[-2])
            target_parts[-1] = int(target_parts[-1])

            pred = result[i][:len(words)]
            term_with_texts_temp = self.terms_from_tags(pred, words)
            term_with_texts = []
            if len(words_real) < len(words):
                for term_with_text in term_with_texts_temp:
                    term_with_text_parts = term_with_text.split('-')
                    if int(term_with_text_parts[-2]) < target_parts[-2]:
                        term_with_texts.append(term_with_text)
                    else:
                        term_with_text_parts[-2] = str(int(term_with_text_parts[-2]) - 2)
                        term_with_text_parts[-1] = str(int(term_with_text_parts[-1]) - 2)
                        term_with_texts.append('-'.join(term_with_text_parts))

                target_parts[-2] = str(target_parts[-2] - 1)
                target_parts[-1] = str(target_parts[-1] - 1)
            else:
                term_with_texts = term_with_texts_temp
                target_parts[-2] = str(target_parts[-2])
                target_parts[-1] = str(target_parts[-1])

            line = json.dumps({'text': text,
                               'pred': term_with_texts,
                               'aspect_terms': ['-'.join(target_parts)]},
                              ensure_ascii=False)
            output_lines.append(line)
        file_utils.write_lines(output_lines, output_filepath)


class AsoTermModel(SequenceLabeling):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.DatasetReaderForAsoTermBiLSTM(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.AsoTermBiLSTM

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.AsoEstimator(self.model, self.val_iterator,
                                                 cuda_device=gpu_id, configuration=self.configuration)
        return estimator

    def evaluate(self):
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data, data_type=data_type)
            self.logger.info('data_type: %s result: %s' % (data_type, result))

    def predict_test(self, output_filepath):
        instances = self.test_data

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.AsoPredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        output_lines = [json.dumps(e, ensure_ascii=False) for e in result]
        file_utils.write_lines(output_lines, output_filepath)

    def predict_test_V2(self, output_filepath):
        reader = self.data_reader

        train_dev_test_data = self.dataset.get_data_type_and_data_dict()

        data_new = []
        for sample in train_dev_test_data['test']:
            sample_new = {
                'words': sample.words,
                'target_tags': sample.target_tags,
                'opinion_words_tags': sample.opinion_words_tags,
                'polarity': sample.polarity,
                'metadata': sample.metadata
            }
            data_new.append(sample_new)

        instances = reader.read(data_new)

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.AsoPredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        output_lines = [json.dumps(e, ensure_ascii=False) for e in result]
        file_utils.write_lines(output_lines, output_filepath)


class AsoTermModelBert(AsoTermModel):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForAsoTermBert(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.AsoTermBiLSTMBert

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                       embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                      # we'll be ignoring masks so we'll need to set this to True
                                                                      allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class AsoBertPair(AsoTermModel):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForAsoBertPair(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.AsoTermBiLSTMBert

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                       embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                      # we'll be ignoring masks so we'll need to set this to True
                                                                      allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class AsoBertPairWithPosition(AsoTermModel):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForAsoBertPairWithPosition(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.AsoTermBiLSTMBertWithPosition

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        from nlp_tasks.absa.mining_opinions.allennlp_bert_supporting_position import bert_token_embedder_supporting_position
        bert_model = bert_token_embedder_supporting_position.PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = bert_token_embedder_supporting_position.BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                       embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                      # we'll be ignoring masks so we'll need to set this to True
                                                                      allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class AsteTermModel(SequenceLabeling):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.AsteEstimator(self.model, self.val_iterator,
                                                 cuda_device=gpu_id, configuration=self.configuration)
        return estimator

    def evaluate(self):
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data, data_type=data_type)
            self.logger.info('data_type: %s result: %s' % (data_type, result))

    def predict_test(self, output_filepath):
        instances = self.test_data

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.AstePredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        predicted_tags = result['predicted_tags']
        sentiment_polarities = result['sentiment_polarities']
        opinion_output_lines = []
        sentiment_output_lines = []
        for i in range(len(instances)):
            instance = instances[i]
            # text = instance.fields['sample'].metadata['metadata']['original_line'].split('####')[0]
            text = instance.fields['sample'].metadata['metadata']['original_line_data']['sentence']
            words_real = text.split(' ')
            words = instance.fields['sample'].metadata['words']

            target_tags = instance.fields['sample'].metadata['target_tags']
            target = self.terms_from_tags(target_tags, words)[0]
            target_parts = target.split('-')
            target_parts[-2] = int(target_parts[-2])
            target_parts[-1] = int(target_parts[-1])

            opinion_tags = predicted_tags[i]
            term_with_texts_temp = self.terms_from_tags(opinion_tags, words)
            term_with_texts = []
            if len(words_real) < len(words):
                for term_with_text in term_with_texts_temp:
                    term_with_text_parts = term_with_text.split('-')
                    if int(term_with_text_parts[-2]) < target_parts[-2]:
                        term_with_texts.append(term_with_text)
                    else:
                        term_with_text_parts[-2] = str(int(term_with_text_parts[-2]) - 2)
                        term_with_text_parts[-1] = str(int(term_with_text_parts[-1]) - 2)
                        term_with_texts.append('-'.join(term_with_text_parts))

                target_parts[-2] = str(target_parts[-2] - 1)
                target_parts[-1] = str(target_parts[-1] - 1)
            else:
                term_with_texts = term_with_texts_temp

            opinion_output_line = json.dumps({'text': text,
                               'pred': term_with_texts,
                               'aspect_terms': ['-'.join(target_parts)]},
                              ensure_ascii=False)
            opinion_output_lines.append(opinion_output_line)

            sentiment = sentiment_polarities[i]
            sentiment_output_line = json.dumps({'text': text,
                                      'aspect_term': '-'.join(target_parts),
                                      'sentiment': sentiment
                                      })
            sentiment_output_lines.append(sentiment_output_line)

        if self.configuration['validation_metric'] in ['+f1', '+opinion_sentiment_f1']:
            file_utils.write_lines(opinion_output_lines, output_filepath + '.opinion')
        if self.configuration['validation_metric'] in ['+sentiment_acc', '+opinion_sentiment_f1']:
            file_utils.write_lines(sentiment_output_lines, output_filepath + '.sentiment')

    def predict_test_v2(self, output_filepath):
        reader = self.data_reader

        train_dev_test_data = self.dataset.get_data_type_and_data_dict()

        data_new = []
        for sample in train_dev_test_data['test']:
            sample_new = {
                'words': sample.words,
                'target_tags': sample.target_tags,
                'opinion_words_tags': sample.opinion_words_tags,
                'polarity': sample.polarity,
                'metadata': sample.metadata
            }
            data_new.append(sample_new)

        instances = reader.read(data_new)

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.AstePredictor(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        predicted_tags = result['predicted_tags']
        sentiment_polarities = result['sentiment_polarities']
        opinion_output_lines = []
        sentiment_output_lines = []
        for i in range(len(instances)):
            instance = instances[i]
            # text = instance.fields['sample'].metadata['metadata']['original_line'].split('####')[0]
            text = instance.fields['sample'].metadata['metadata']['original_line_data']['sentence']
            words_real = text.split(' ')
            words = instance.fields['sample'].metadata['words']

            target_tags = instance.fields['sample'].metadata['target_tags']
            target = self.terms_from_tags(target_tags, words)[0]
            target_parts = target.split('-')
            target_parts[-2] = int(target_parts[-2])
            target_parts[-1] = int(target_parts[-1])

            opinion_tags = predicted_tags[i]
            term_with_texts_temp = self.terms_from_tags(opinion_tags, words)
            term_with_texts = []
            if len(words_real) < len(words):
                for term_with_text in term_with_texts_temp:
                    term_with_text_parts = term_with_text.split('-')
                    if int(term_with_text_parts[-2]) < target_parts[-2]:
                        term_with_texts.append(term_with_text)
                    else:
                        term_with_text_parts[-2] = str(int(term_with_text_parts[-2]) - 2)
                        term_with_text_parts[-1] = str(int(term_with_text_parts[-1]) - 2)
                        term_with_texts.append('-'.join(term_with_text_parts))

                target_parts[-2] = str(target_parts[-2] - 1)
                target_parts[-1] = str(target_parts[-1] - 1)
            else:
                term_with_texts = term_with_texts_temp

            opinion_output_line = json.dumps({'text': text,
                               'pred': term_with_texts,
                               'aspect_terms': ['-'.join(target_parts)]},
                              ensure_ascii=False)
            opinion_output_lines.append(opinion_output_line)

            sentiment = sentiment_polarities[i]
            sentiment_output_line = json.dumps({'text': text,
                                      'aspect_term': '-'.join(target_parts),
                                      'sentiment': sentiment
                                      })
            sentiment_output_lines.append(sentiment_output_line)

        if self.configuration['validation_metric'] in ['+f1', '+opinion_sentiment_f1']:
            file_utils.write_lines(opinion_output_lines, output_filepath + '.opinion')
        if self.configuration['validation_metric'] in ['+sentiment_acc', '+opinion_sentiment_f1']:
            file_utils.write_lines(sentiment_output_lines, output_filepath + '.sentiment')


class WarmupAsteTermModel(AsteTermModel):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_optimizer(self, model: pytorch_models.WarmupSequenceLabelingModel, fine_tuning_towe=False):
        lr = 0.001
        if fine_tuning_towe:
            towe_layers = model.towe_layers()
            towe_parameters = []
            for layer in towe_layers:
                for parameter in layer.parameters():
                    if parameter.requires_grad:
                        towe_parameters.append(parameter)
            ignored_params = list(map(id, towe_parameters))  # parameters
            base_params = filter(lambda p: p.requires_grad and id(p) not in ignored_params, model.parameters())
            return optim.Adam(
                [{'params': base_params},
                 {'params': towe_parameters, 'lr': lr * self.configuration['learning_rate_scale_for_fine_tuning_towe']}],
                lr=lr, weight_decay=0.00001)
        else:
            _params = filter(lambda p: p.requires_grad, model.parameters())
            return optim.Adam(_params, lr=lr, weight_decay=0.00001)

    def _inner_train(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1

        self.model: pytorch_models.WarmupSequenceLabelingModel = self._find_model_function()

        estimator = self._get_estimator(self.model)

        if self.configuration['towe_warmup']:
            optimizer = self._get_optimizer(self.model)
            self.logger.info('towe_warmup')
            validation_metric = '+f1'
            callbacks = self._get_estimate_callback(self.model)
            loss_weights = {
                'atsa_loss_weight': 0,
                'towe_loss_weight': 1
            }
            callbacks.extend(self._get_fixed_loss_weight_callback(self.model, loss_weights))
            self._print_args(self.model)
            towe_model_dir = os.path.join(self.model_dir, 'towe')
            if not os.path.exists(towe_model_dir):
                os.makedirs(towe_model_dir)

            trainer = Trainer(
                model=self.model,
                optimizer=optimizer,
                iterator=self.iterator,
                train_dataset=self.train_data,
                validation_dataset=self.dev_data,
                cuda_device=gpu_id,
                num_epochs=self.configuration['epochs'],
                validation_metric=validation_metric,
                validation_iterator=self.val_iterator,
                serialization_dir=towe_model_dir,
                patience=self.configuration['patience'],
                callbacks=callbacks,
                num_serialized_models_to_keep=0,
                early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
                estimator=estimator,
                grad_clipping=5
            )
            metrics = trainer.train()
            self.logger.info('towe metrics: %s' % str(metrics))

        validation_metric = '+f1'
        if 'validation_metric' in self.configuration:
            validation_metric = self.configuration['validation_metric']
        self.logger.info('validation_metric: %s' % validation_metric)

        callbacks = self._get_estimate_callback(self.model)
        loss_weights = {
            'atsa_loss_weight': self.configuration['atsa_loss_weight'],
            'towe_loss_weight': self.configuration['towe_loss_weight']
        }
        callbacks.extend(self._get_fixed_loss_weight_callback(self.model, loss_weights))

        fine_tuning_towe = False
        if self.configuration['towe_warmup'] and self.configuration['fine_tune_towe']:
            fine_tuning_towe = True
        optimizer = self._get_optimizer(self.model, fine_tuning_towe=fine_tuning_towe)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            iterator=self.iterator,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            cuda_device=gpu_id,
            num_epochs=self.configuration['epochs'],
            validation_metric=validation_metric,
            validation_iterator=self.val_iterator,
            serialization_dir=self.model_dir,
            patience=self.configuration['patience'],
            callbacks=callbacks,
            num_serialized_models_to_keep=0,
            early_stopping_by_batch=self.configuration['early_stopping_by_batch'],
            estimator=estimator,
            grad_clipping=5
        )
        metrics = trainer.train()
        self.logger.info('metrics: %s' % str(metrics))


class AsteTermBiLSTM(WarmupAsteTermModel):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        if self.configuration['aspect_term_aware']:
            reader = sequence_labeling_data_reader.DatasetReaderForAsteTermBiLSTM(
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                configuration=self.configuration
            )
        else:
            reader = sequence_labeling_data_reader.DatasetReaderForAsteTermBiLSTMWithoutSpecialToken(
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                configuration=self.configuration
            )
        return reader

    def _find_model_function_pure(self):
        if self.configuration['sequence_label_attention']:
            return pytorch_models.AsteTermBiLSTMWithSLA
        else:
            return pytorch_models.AsteTermBiLSTM

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class MILForAso(WarmupAsteTermModel):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.DatasetReaderForMilAsoTermBiLSTM(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def _load_data(self):
        reader = self._get_data_reader()
        self.data_reader = reader

        data_filepath = self.base_data_dir + 'data'
        if os.path.exists(data_filepath):
            self.train_data, self.dev_data, self.test_data, = super()._load_object(data_filepath)
        else:
            train_dev_test_data = self.dataset.get_data_type_and_data_dict()

            train_dev_test_data_new = {}
            for data_type, data in train_dev_test_data.items():
                data_new = []
                for sample in data:
                    if not self.configuration['train_mil_with_conflict'] and (data_type == 'train' or data_type == 'dev') and sample.polarity == 'conflict':
                        continue
                    sample_new = {
                        'words': sample.words,
                        'target_tags': sample.target_tags,
                        'opinion_words_tags': sample.opinion_words_tags,
                        'polarity': sample.polarity,
                        'metadata': sample.metadata
                    }
                    data_new.append(sample_new)
                train_dev_test_data_new[data_type] = data_new

            self.train_data = reader.read(train_dev_test_data_new['train'])
            self.dev_data = reader.read(train_dev_test_data_new['dev'])
            self.test_data = reader.read(train_dev_test_data_new['test'])
            data = [self.train_data, self.dev_data, self.test_data]
            super()._save_object(data_filepath, data)

    def _find_model_function_pure(self):
        return pytorch_models.MILForASO

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.MilAsoEstimator(self.model,
                                                   self.val_iterator,
                                                   cuda_device=gpu_id,
                                                   configuration=self.configuration)
        return estimator

    def evaluate(self):
        estimator = self._get_estimator(self.model)

        data_type_and_data = {
            'train': self.train_data,
            'dev': self.dev_data,
            'test': self.test_data
        }
        for data_type, data in data_type_and_data.items():
            result = estimator.estimate(data, data_type=data_type)
            self.logger.info('data_type: %s result: %s' % (data_type, result))

    def predict_test(self, output_filepath):
        instances = self.test_data

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.MilAsoPredictor(self.model, self.val_iterator,
                                                   cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        output_lines = [json.dumps(e, ensure_ascii=False) for e in result]
        file_utils.write_lines(output_lines, output_filepath)

    def predict_test_V2(self, output_filepath):
        case_words = [
            ["The", "atmosphere", "is", "attractive", ",", "but", "a", "little", "uncomfortable", "."],
            ["The", "service", "is", "a", "quite", "slow", ",", "but", "friendly", "."]
        ]

        reader = self.data_reader

        train_dev_test_data = self.dataset.get_data_type_and_data_dict()

        data_new = []
        for sample in train_dev_test_data['test']:

            # if sample.words not in case_words:
            #     continue

            sample_new = {
                'words': sample.words,
                'target_tags': sample.target_tags,
                'opinion_words_tags': sample.opinion_words_tags,
                'polarity': sample.polarity,
                'metadata': sample.metadata
            }
            data_new.append(sample_new)

        instances = reader.read(data_new)

        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        predictor = pytorch_models.MilAsoPredictor(self.model, self.val_iterator,
                                                   cuda_device=gpu_id, configuration=self.configuration)

        result = predictor.predict(instances)
        output_lines = [json.dumps(e, ensure_ascii=False) for e in result]
        file_utils.write_lines(output_lines, output_filepath)


class MILForAsoBert(MILForAso):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):

        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForMilAsoTermBert(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.MILForASOBert

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function_pure(self):
        return pytorch_models.MILForASOBert

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                       embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                      # we'll be ignoring masks so we'll need to set this to True
                                                                      allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()
        another_bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder,
            another_bert_word_embedder=another_bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model, fine_tuning_towe=False):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])


class TermBert(ToweModel):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForTermBert(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])

    def _find_model_function_pure(self):
        return pytorch_models.TermBert

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_estimator(self, model):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            gpu_id = self.configuration['gpu_id']
        else:
            gpu_id = -1
        estimator = pytorch_models.ToweEstimator(self.model, self.val_iterator,
                                                                  cuda_device=gpu_id, configuration=self.configuration)
        return estimator


class AsteTermBert(WarmupAsteTermModel):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        if self.configuration['aspect_term_aware']:
            reader = sequence_labeling_data_reader.DatasetReaderForAsteTermBert(
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                configuration=self.configuration,
                bert_tokenizer=bert_tokenizer,
                bert_token_indexers={"bert": bert_token_indexer}
            )
        else:
            reader = sequence_labeling_data_reader.DatasetReaderForAsteTermBertWithoutSpecialToken(
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                configuration=self.configuration,
                bert_tokenizer=bert_tokenizer,
                bert_token_indexers={"bert": bert_token_indexer}
            )
        return reader

    def _get_optimizer(self, model: pytorch_models.WarmupSequenceLabelingModel, fine_tuning_towe=False):
        if self.configuration['fixed_bert']:
            return super()._get_optimizer(model, fine_tuning_towe=fine_tuning_towe)
        else:
            lr = self.configuration['learning_rate_in_bert']
            weight_decay = self.configuration['l2_in_bert']

            if fine_tuning_towe:
                towe_layers = model.towe_layers()
                towe_parameters = []
                for layer in towe_layers:
                    for parameter in layer.parameters():
                        if parameter.requires_grad:
                            towe_parameters.append(parameter)
                ignored_params = list(map(id, towe_parameters))  # parameters
                base_params = filter(lambda p: p.requires_grad and id(p) not in ignored_params, model.parameters())
                return optim.Adam(
                    [{'params': base_params},
                     {'params': towe_parameters,
                      'lr': lr * self.configuration['learning_rate_scale_for_fine_tuning_towe']}],
                    lr=lr, weight_decay=weight_decay)
            else:
                _params = filter(lambda p: p.requires_grad, model.parameters())
                return optim.Adam(_params, lr=lr, weight_decay=weight_decay)

    def _find_model_function_pure(self):
        if self.configuration['sequence_label_attention']:
            return pytorch_models.AsteTermBertWithSLA
        else:
            return pytorch_models.AsteTermBert

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        if self.configuration['sequence_label_attention']:
            another_bert_word_embedder = self._get_bert_word_embedder()
            model = model_function(
                word_embedder,
                position_embedder,
                self.vocab,
                self.configuration,
                bert_word_embedder=bert_word_embedder,
                another_bert_word_embedder = another_bert_word_embedder
            )
        else:
            model = model_function(
                word_embedder,
                position_embedder,
                self.vocab,
                self.configuration,
                bert_word_embedder=bert_word_embedder
            )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model


class TermBertWithSecondSentence(TermBert):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForTermBertWithSecondSentence(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader


class TermBertWithSecondSentenceWithPosition(TermBert):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForTermBertWithSecondSentenceWithPosition(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        from nlp_tasks.absa.mining_opinions.allennlp_bert_supporting_position import \
            bert_token_embedder_supporting_position
        bert_model = bert_token_embedder_supporting_position.PretrainedBertModel.load(pretrained_model,
                                                                                      cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = bert_token_embedder_supporting_position.BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function_pure(self):
        return pytorch_models.TermBertWithPosition


class TermBiLSTMWithSecondSentence(TermBert):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForTermBiLSTMWithSecondSentence(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        from nlp_tasks.absa.mining_opinions.allennlp_bert_supporting_position import \
            bert_token_embedder_supporting_position
        bert_model = bert_token_embedder_supporting_position.PretrainedBertModel.load(pretrained_model,
                                                                                      cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = bert_token_embedder_supporting_position.BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function_pure(self):
        return pytorch_models.TermBiLSTMWithSecondSentence


class AsteTermBertWithSecondSentence(AsteTermBert):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForAsteTermBertWithSecondSentence(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader


class NerLstm(SequenceLabeling):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        if 'RealASO' in self.configuration['current_dataset']:
            reader = sequence_labeling_data_reader.DatasetReaderForNerLstmOfRealASO(
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                configuration=self.configuration
            )
        else:
            reader = sequence_labeling_data_reader.DatasetReaderForNerLstm(
                tokenizer=self._get_word_segmenter(),
                token_indexers={"tokens": token_indexer},
                position_indexers={'position': position_indexer},
                configuration=self.configuration
            )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.NerLstm

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class NerLstmForOTE(NerLstm):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.DatasetReaderForNerLstmForOTEOfRealASO(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.NerLstm

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class NerBertForOTE(NerLstmForOTE):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForNerBertForOTEOfRealASO(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _get_bert_word_embedder(self):
        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function_pure(self):
        return pytorch_models.NerBertForOTE

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class NerBert(SequenceLabeling):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForNerBert(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function_pure(self):
        return pytorch_models.NerBert

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class IOG(ToweModel):
    """

    """

    def __init__(self, configuration):
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        reader = sequence_labeling_data_reader.DatasetReaderForIOG(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration
        )
        return reader

    def _find_model_function_pure(self):
        return pytorch_models.IOG

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(_params)
