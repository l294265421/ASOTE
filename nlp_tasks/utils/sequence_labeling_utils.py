from typing import List


def first_term_from_tags(tags: List[str], start_index: int):
    """

    :param tags:
    :param start_index:
    :return:
    """
    if 'B' in tags[start_index:]:
        start_index = tags.index('B', start_index)
        end_index = start_index + 1
        while end_index < len(tags) and tags[end_index] == 'I':
            end_index += 1
        return [start_index, end_index]
    else:
        return None


def terms_from_tags(tags: List[str], words: List[str]):
    """
    words = ['We', 'went', 'to', 'eat', 'at', 'the', 'Jekyll', 'and', 'Hyde', 'restaurant', 'on', 'Friday', 'night', 'and', 'really', 'enjoyed', 'the', 'fun', 'atmosphere', 'and', 'good', 'food', '.']
    tags = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'B', 'O', 'O']

    :param tags:
    :return: ['enjoyed-15-16', 'good-20-21']
    """
    tags = tags[: len(words)]

    terms = []
    start_index = 0
    while start_index < len(tags):
        term = first_term_from_tags(tags, start_index)
        if term is None:
            break
        else:
            terms.append(term)
            start_index = term[1]

    term_with_texts = []
    for term in terms:
        term_text = ' '.join(words[term[0]: term[1]])
        term_with_texts.append('%s-%d-%d' % (term_text, term[0], term[1]))
    return term_with_texts


if __name__ == '__main__':
    words = ['We', 'went', 'to', 'eat', 'at', 'the', 'Jekyll', 'and', 'Hyde', 'restaurant', 'on', 'Friday', 'night', 'and', 'really', 'enjoyed', 'the', 'fun', 'atmosphere', 'and', 'good', 'food', '.']
    tags = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'B', 'O', 'O']
    result = terms_from_tags(tags, words)
    print(result)