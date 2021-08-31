# -*- coding: utf-8 -*-
#


from sklearn.feature_extraction import text as sklearn_text
from nlp_tasks.utils import tokenizers


def get_trained_count_and_tfidf_model(texts: list, tokenizer: tokenizers.BaseTokenizer()):
    """

    :param texts:
    :param tokenizer:
    :return:
    """
    texts_tokenized = [' '.join(tokenizer(text)) for text in texts]
    # ：a[i][j] ji
    count_model = sklearn_text.CountVectorizer()
    count_model.fit(texts_tokenized)
    freq_word_matrix = count_model.transform(texts_tokenized)

    # tf-idf
    tfidf_model = sklearn_text.TfidfTransformer()
    tfidf_model.fit(freq_word_matrix)

    return count_model, tfidf_model


def to_tfidf_vectors3(texts: list, count_model, tfidf_model):
    """

    :param texts:
    :param tokenizer:
    :param count_model:
    :param tfidf_model:
    :return:
    """
    count = count_model.transform(texts)
    result = tfidf_model.transform(count)
    return result


def to_tfidf_vectors2(texts: list, tokenizer: tokenizers.BaseTokenizer(), count_model, tfidf_model):
    """

    :param texts:
    :param tokenizer:
    :param count_model:
    :param tfidf_model:
    :return:
    """
    texts_tokenized = [' '.join(tokenizer(text)) for text in texts]
    return to_tfidf_vectors3(texts_tokenized, count_model, tfidf_model)


def to_tfidf_vectors(texts: list, tokenizer: tokenizers.BaseTokenizer()):
    """

    :param texts: list of str
    :param tokenizer:
    :return: ，text
    """
    texts_tokenized = [' '.join(tokenizer(text)) for text in texts]
    # ：a[i][j] ji
    vectorizer = sklearn_text.CountVectorizer()
    freq_word_matrix = vectorizer.fit_transform(texts_tokenized)

    # tf-idf
    transformer = sklearn_text.TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(freq_word_matrix)

    X = tfidf_matrix.toarray()
    return X


if __name__ == '__main__':
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
    ]
    vectorizer, transformer = get_trained_count_and_tfidf_model(corpus, tokenizer=tokenizers.BaseTokenizer())
    # vectorizer = CountVectorizer()
    # count = vectorizer.fit_transform(corpus)
    corpus = [
        'This is the first document.',
        'This is the second second document.',
    ]
    count = vectorizer.transform(corpus)
    print(vectorizer.get_feature_names())
    print(vectorizer.vocabulary_)
    print(count.toarray())

    # transformer = TfidfTransformer()
    # transformer.fit(count)
    tfidf_matrix = transformer.transform(count)
    print(tfidf_matrix.toarray())