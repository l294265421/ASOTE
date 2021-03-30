# -*- coding: utf-8 -*-


import re


class ZhSplitStentence(object):
    """
    句子切分
    """

    def __init__(self,
                 cal_offset: bool = False,
                 pattern: str = "[。？?！!]"):

        self._cal_offset = cal_offset
        self._pattern = pattern
        self._complie_pattern = re.compile(self._pattern)

    def __call__(self, text: str):

        end_tags = self._complie_pattern.findall(text)

        sentences = list()

        start = 0
        for end_tag in end_tags:
            index = text.index(end_tag, start)

            sentence = text[start: index+1].strip()

            if len(sentence) > 0:
                sentences.append(sentence)
            start = index + 1
        if not sentences:
            sentences.append(text)
        return sentences


class ZhSplitParagraph(object):
    """
    段落
    """

    def __init__(self,
                 cal_offset: bool = False,
                 pattern: str = "\r\n"):

        self._cal_offset = cal_offset
        self._pattern = pattern
        self._complie_pattern = re.compile(self._pattern)

    def __call__(self, text: str):
        if not text.endswith('\r\n'):
            text += '\r\n'
        end_tags = self._complie_pattern.findall(text)

        sentences = list()

        start = 0
        for end_tag in end_tags:
            index = text.index(end_tag, start)

            sentence = text[start: index+1].strip()

            if len(sentence) > 0:
                sentences.append(sentence)
            start = index + 1
        if not sentences:
            sentences.append(text)
        return sentences
