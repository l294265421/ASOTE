import abc

import nltk
from nltk.corpus import stopwords

stemmer = nltk.stem.SnowballStemmer('english')
english_stop_words = stopwords.words('english')


class WordProcessorInterface:
    """

    """

    @abc.abstractmethod
    def process(self, word: str):
        pass

    @abc.abstractmethod
    def get_description(self):
        pass


class BaseWordProcessor(WordProcessorInterface):
    """

    """

    def __init__(self, other_processor: WordProcessorInterface=None):
        self.other_processor = other_processor

    def get_description(self):
        """

        :return:
        """
        result = ''
        if not self.other_processor is None:
            result += self.other_processor.get_description()
            result + '->'
        return result + self.__class__.__name__

    def _inner_process(self, word: str):
        """

        :return:
        """
        return word

    def process(self, word: str):
        """

        :param word:
        :return:
        """
        if word is None:
            return None
        if self.other_processor is not None:
            word = self.other_processor.process(word)
        return self._inner_process(word)


class LowerProcessor(BaseWordProcessor):
    """

    """

    def __init__(self, other_processor: WordProcessorInterface=None):
        super().__init__(other_processor)

    def _inner_process(self, word: str):
        """

        :param word:
        :return:
        """
        return word.lower()


class StemProcessor(BaseWordProcessor):
    """

    """

    def __init__(self, other_processor: WordProcessorInterface=None):
        super().__init__(other_processor)

    def _inner_process(self, word: str):
        """

        :param word:
        :return:
        """
        return stemmer.stem(word)


class StopWordProcessor(BaseWordProcessor):
    """

    """

    def __init__(self, other_processor: WordProcessorInterface = None):
        super().__init__(other_processor)

    def _inner_process(self, word: str):
        """

        :param word:
        :return:
        """
        if word in english_stop_words:
            return None
        else:
            return word
