# -*- coding: utf-8 -*-



def split_sentence(sentence: str, length: int=100):
    """
    把太长的句子拆分成多行，方便在标注工具中观察
    :param sentence:
    :param length:
    :return:
    """

    result = []
    start = 0
    while start < len(sentence):
        result.append(sentence[start: start + length])
        start += length
    return result



