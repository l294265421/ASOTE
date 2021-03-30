# -*- coding: utf-8 -*-


from nlp_tasks.utils import my_corenlp
from nlp_tasks.utils import corenlp_factory


class CorenlpParser:
    def __init__(self, nlp: my_corenlp.StanfordCoreNLP, cache_sentence_parse_result=False,
                 max_cache_sentence_num=300):
        self.nlp = nlp
        self.cache_sentence_parse_result = cache_sentence_parse_result
        self.sentence_parse_result = {}
        self.cache_keys = []
        self.max_cache_sentence_num = max_cache_sentence_num

    def build_parse_child_dict(self, postags, arcs):
        """
        '句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典
        :param postags:
        ('以及', 'CC'), ('公司', 'NN'), ('内部', 'NN'), ('的', 'DEG'), ('党派', 'NN'), ('之', 'DEG'), ('争', 'NN'),
        ('最终', 'AD'), ('导致', 'VV'), ('了', 'AS'), ('尚', 'AD'), ('阳', 'NR'), ('科技', 'NN'), ('在', 'P'),
        ('2006年', 'NT'), ('梦', 'NN'), ('碎', 'VA'), ('当场', 'AD'), ('。', 'PU')
        :param arcs:
        [('ROOT', 0, 1), ('compound:nn', 3, 2), ('nmod:assmod', 5, 3), ('case', 3, 4), ('nmod:assmod', 7, 5),
        ('case', 5, 6), ('nsubj', 9, 7), ('advmod', 9, 8), ('root', 1, 9), ('aux:asp', 9, 10), ('advmod', 17, 11),
        ('compound:nn', 13, 12), ('nsubj', 17, 13), ('case', 16, 14), ('compound:nn', 16, 15), ('nmod:prep', 17, 16),
        ('ccomp', 9, 17), ('advmod', 17, 18), ('punct', 9, 19)]
        :return:
        format_parse_list
         [['SBV', '提高', 0, 'v', '导致', 2, 'v'], ['VOB', '关税', 1, 'n', '提高', 0, 'v'],
        ['HED', '导致', 2, 'v', 'Root', -1, 'wp'], ['ATT', '国际', 3, 'n', '秩序', 5, 'n'],
        ['ATT', '贸易', 4, 'v', '秩序', 5, 'n'], ['SBV', '秩序', 5, 'n', '混乱', 6, 'a'],
        ['VOB', '混乱', 6, 'a', '导致', 2, 'v'], ['WP', '，', 7, 'wp', '导致', 2, 'v']]
        ['SBV', '提高', 0, 'v', '导致', 2, 'v'] 每一个元素的含义分别是 关系类型，当前词，当前词位置，当前词词性，
        依赖词，依赖词位置，和依赖词词性
        child_dict_list
        [{'VOB': [1]}, {}, {'SBV': [0], 'VOB': [6], 'WP': [7]}, {}, {}, {'ATT': [3, 4]}, {'SBV': [5]}, {}]
        {'VOB': [1]} key为关系类型，value为与该词存在此种关系的词的位置
        """
        # 谁指向自己，关系是什么
        format_parse_list = [[] for _ in range(len(postags))]
        # 自己指向了谁，关系是什么
        child_dict_list = [{} for _ in range(len(postags))]
        for element in arcs:
            relation, start_index, end_index = element
            start_index -= 1
            end_index -= 1
            current_word = postags[end_index][0]
            current_word_pos = postags[end_index][1]
            head_word = postags[start_index][0] if start_index != -1 else 'ROOT'
            head_word_pos = postags[start_index][1] if start_index != -1 else ''
            format_parse_list[end_index] = [relation, current_word, end_index, current_word_pos,
                                            head_word, start_index, head_word_pos]
            if start_index == -1:
                continue
            if relation not in child_dict_list[start_index]:
                child_dict_list[start_index][relation] = []
            child_dict_list[start_index][relation].append(end_index)
        return child_dict_list, format_parse_list

    def _inner_parser_main(self, sentence):
        # words = self.nlp.word_tokenize(sentence)
        postags = self.nlp.pos_tag(sentence)
        words = [e[0] for e in postags]
        arcs = self.nlp.dependency_parse(sentence)
        child_dict_list, format_parse_list = self.build_parse_child_dict(postags, arcs)
        return words, postags, arcs, child_dict_list, format_parse_list

    def parser_main(self, sentence):
        """
        parser主函数
        :param sentence:
        :return:
        """
        if self.cache_sentence_parse_result:
            dependency_key = ('%s-dependency' % sentence)
            if dependency_key in self.sentence_parse_result:
                words, postags, arcs, child_dict_list, format_parse_list = self.sentence_parse_result[dependency_key]
                return words, postags, arcs, child_dict_list, format_parse_list
            else:
                words, postags, arcs, child_dict_list, format_parse_list = self._inner_parser_main(sentence)
                self.sentence_parse_result[dependency_key] = [words, postags, arcs, child_dict_list, format_parse_list]
                self.cache_keys.append(dependency_key)
                if len(self.cache_keys) > self.max_cache_sentence_num:
                    cache_key = self.cache_keys.pop(0)
                    self.sentence_parse_result.pop(cache_key)
                return words, postags, arcs, child_dict_list, format_parse_list
        else:
            words, postags, arcs, child_dict_list, format_parse_list = self._inner_parser_main(sentence)
            return words, postags, arcs, child_dict_list, format_parse_list


if __name__ == '__main__':
    parser = CorenlpParser(corenlp_factory.create_corenlp_server())
    # sentence = sys.argv[1]
    sentence = '以及公司内部的党派之争最终导致了尚阳科技在2006年梦碎当场。'
    words, postags, arcs, child_dict_list, format_parse_list = parser.parser_main(sentence)
    print(postags, len(postags))
    print(arcs, len(arcs))
    print(child_dict_list, len(child_dict_list))
    print(format_parse_list, len(format_parse_list))
