"""

"""

import os
import logging
import sys
import pickle

import numpy as np

from nlp_tasks.common import common_path
from nlp_tasks.utils import file_utils
from nlp_tasks.absa.data_adapter import data_object
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
        if self.configuration['current_dataset'] in ['SemEval-2016-Task-5-CH-CAME-SB1',
                                                     'SemEval-2016-Task-5-CH-PHNS-SB1']:
            word_segmenter = tokenizers.JiebaTokenizer()
        else:
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
