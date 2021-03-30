import re


def preprocess(text: str):
    """

    :param text:
    :return:
    """
    text_clean = text.replace('\n', '')
    text_clean = text_clean.lower()
    return text_clean


if __name__ == '__main__':
    text = '\nMost everything is fine with this machine: speed, capacity, build.\n'
    print(preprocess(text))
    print(text)
