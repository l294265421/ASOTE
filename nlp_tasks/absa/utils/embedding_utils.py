import numpy as np


def get_coefs(word, *arr):
    try:
        if len(arr) > 300:
            word += ''.join(arr[-300:])
            arr = arr[-300:]
        return word, np.asarray(arr, dtype='float32')
    except:
        print(word)
        print(arr)


def generate_word_embedding(embedding_file_path, words):
    word_embedding = dict(get_coefs(*o.strip().split()) for o in open(embedding_file_path, encoding='utf-8'))
    result = {}
    for word in words:
        if word in word_embedding:
            result[word] = word_embedding[word]
        else:
            result[word] = np.random.uniform(-0.25, 0.25, task_conf.embed_size)
    return result


def generate_word_embedding_all(embedding_file_path):
    word_embedding = dict(get_coefs(*o.strip().split()) for o in open(embedding_file_path, encoding='utf-8'))
    return word_embedding


def generate_embedding_matrix(word_index, nb_words, embedding_file_path, embed_size):
    """

    Args:
        word_index: dict, key: 词 value: 词在词典中索引位置
        nb_words: int, 词典大小
        embedding_file_path: embedding文件路径
        embed_size: 词向量长度
    """

    word_embedding = generate_word_embedding(embedding_file_path, word_index.keys())
    all_embs = np.stack(word_embedding.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = word_embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
