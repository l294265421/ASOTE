from typing import List
import re

import nltk


class BaseSentenceSegmenter:
    """
    切分句子
    """

    def __init__(self, configuration: dict=None):
        self.configuration = configuration

    def sent_tokenize(self, text: str):
        result = []
        paragraphs = text.splitlines()
        for paragraph in paragraphs:
            sentences = self._inner_sent_tokenize(paragraph)
            result.extend(sentences)
        return result

    def _inner_sent_tokenize(self, line: str) -> List[str]:
        pass


class NltkSentenceSegmenter(BaseSentenceSegmenter):
    """

    """

    def __init__(self, configuration: dict=None):
        super().__init__(configuration)

    def _inner_sent_tokenize(self, line: str) -> List[str]:
        return nltk.sent_tokenize(line)


class ConstituencyParseSentenceSegmenter(BaseSentenceSegmenter):
    """

    """

    def __init__(self, configuration: dict=None):
        super().__init__(configuration)

    def _inner_sent_tokenize(self, line: str) -> List[str]:
        return nltk.sent_tokenize(line)


class SimpleChineseSentenceSegmenter(BaseSentenceSegmenter):
    """

    """

    def __init__(self, configuration: dict=None):
        super().__init__(configuration)

    def _inner_sent_tokenize(self, line: str) -> List[str]:
        # 。+|；|！+|？+|
        pattern = '([^。；！？]+)(。+|；|！+|？+|$)'
        all = re.findall(pattern, line)
        result = []
        for e in all:
            sentence = ''.join(e)
            result.append(sentence)
        return result
