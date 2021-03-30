# -*- coding: utf-8 -*-
"""
通过训练数据集中有的词从词向量全集中生成一个只包含这些词的词向量子集文件；处于两点考虑:
1. 减少词向量文件的大小
2. 在一个词向量文件中没有的词，可以通过另一个词向量文件补充
Date:    2018/9/30 13:07
"""

import sys

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.utils import embedding_utils


def __get_words(feature_lines):
    words = set()
    for feature_line in feature_lines:
        words = words.union(feature_line.split())
    return words


if __name__ == '__main__':
    train_words = data_utils.read_features(data_path.train_subject_word_file_path)
    val_words = data_utils.read_features(data_path.val_subject_word_file_path)
    test_words = data_utils.read_features(data_path.test_subject_word_file_path)

    all_words = set()
    all_words = all_words.union(__get_words(train_words))
    all_words = all_words.union(__get_words(val_words))
    all_words = all_words.union(__get_words(test_words))

    file_utils.write_lines(all_words, data_path.data_base_dir + data_path.all_word_file_path)

    embedding_file_path = sys.argv[1:]
    word_embedding = {}
    miss_embedding_words = all_words
    print(len(miss_embedding_words))
    for embedding_file_path_sample in embedding_file_path:
        temp = embedding_utils.generate_word_embedding(
            data_path.embedding_base_dir + embedding_file_path_sample, miss_embedding_words)
        word_embedding.update(temp)
        miss_embedding_words = miss_embedding_words.difference(temp.keys())
        print(len(miss_embedding_words))

    word_embedding_lines = []
    for word, embedding in word_embedding.items():
        embedding_str = [str(element) for element in embedding]
        word_embedding_lines.append(word + ' ' + ' '.join(embedding_str))
    file_utils.write_lines(word_embedding_lines, data_path.embedding_file)
