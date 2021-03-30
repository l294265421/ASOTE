# -*- coding: utf-8 -*-


from nlp_tasks.utils import tokenizers


def to_english_like_sentence(sentence: str, tokenizer = tokenizers.JiebaTokenizer()):
    """

    :param sentence: 我在百度
    :return: 我 在 百度
    """
    return ' '.join(tokenizer(sentence))
