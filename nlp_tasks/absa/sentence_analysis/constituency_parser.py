"""
https://spacy.io/universe/project/self-attentive-parser
https://github.com/nikitakit/self-attentive-parser
https://parser.kitaev.io/
"""

import logging
from typing import List
import queue

import spacy
from spacy.language import Language
from spacy.tokens.span import Span
# from benepar.spacy_plugin import BeneparComponent
import matplotlib.pyplot as plt
import networkx as nx


class SerialNumber:
    """

    """

    def __init__(self, current_number=0):
        self.current_number = current_number

    def next(self):
        result = self.current_number
        self.current_number += 1
        return result


class ConstituencyTreeNode:
    """

    """
    def __init__(self, labels: str, text: str, node_id: int = -1, start: int = -1, end: int = -1,
                 depth: int = -1):
        self.labels = labels
        self.text = text
        self.node_id = node_id
        self.start = start
        self.end = end
        self.parent: ConstituencyTreeNode = None
        self.children: List[ConstituencyTreeNode] = []
        self.metadata: dict = {}
        self.depth = depth

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def get_adjacency_list(self) -> List[List['ConstituencyTreeNode']]:
        result = []
        helper_queue = queue.Queue()
        helper_queue.put(self)
        while not helper_queue.empty():
            cursor: ConstituencyTreeNode = helper_queue.get()
            for child in cursor.children:
                result.append([cursor, child])
                helper_queue.put(child)
        return result

    @staticmethod
    def get_all_leaves(node: 'ConstituencyTreeNode'):
        """

        :param node:
        :return:
        """
        result = []
        for child in node.children:
            if child.start + 1 == child.end:
                result.append(child)
            result.extend(ConstituencyTreeNode.get_all_leaves(child))
        return result

    @staticmethod
    def get_all_inner_nodes(node: 'ConstituencyTreeNode'):
        """

        :param node:
        :return:
        """
        result = []
        if node.start + 1 != node.end:
            result.append(node)
        for child in node.children:
            result.extend(ConstituencyTreeNode.get_all_inner_nodes(child))
        return result

    @staticmethod
    def get_all_nodes(node: 'ConstituencyTreeNode'):
        """

        :param node:
        :return:
        """
        result = [node]
        for child in node.children:
            result.extend(ConstituencyTreeNode.get_all_nodes(child))
        return result

    def get_adjacency_list_between_all_node_and_leaf(self) -> List[List['ConstituencyTreeNode']]:
        result = []
        helper_queue = queue.Queue()
        helper_queue.put(self)
        while not helper_queue.empty():
            cursor: ConstituencyTreeNode = helper_queue.get()
            for child in cursor.children:
                helper_queue.put(child)
            for offspring in ConstituencyTreeNode.get_all_leaves(cursor):
                result.append([cursor, offspring])
        return result

    @staticmethod
    def parse_using_spacy(spacy_nlp: Language, sentence: str) -> 'ConstituencyTreeNode':
        """
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(BeneparComponent('benepar_en'))
        :param spacy_nlp:
        :param sentence:
        :return:
        """
        doc = spacy_nlp(sentence)
        sents = list(doc.sents)
        if len(sents) > 1:
            trees = []
            words_num = sents[-1].end
            serial_number = SerialNumber(current_number=words_num)
            for sent in sents:
                tree = ConstituencyTreeNode._inner_parse_using_spacy(sent, serial_number, 1)
                trees.append(tree)

            labels = ('S',)
            start = sents[0].start
            end = sents[-1].end
            text = ' '.join([sent.text for sent in sents])
            node_id = serial_number.next()
            result = ConstituencyTreeNode(labels, text, node_id=node_id, start=start, end=end, depth=0)
            result.children = trees
        else:
            sent = sents[0]
            words_num = sent.end
            serial_number = SerialNumber(current_number=words_num)
            result = ConstituencyTreeNode._inner_parse_using_spacy(sent, serial_number, 0)
        return result

    @staticmethod
    def _inner_parse_using_spacy(sentence: Span, serial_number: SerialNumber, depth) -> 'ConstituencyTreeNode':
        labels = sentence._.labels
        start = sentence.start
        end = sentence.end
        text = sentence.text
        if start + 1 == end:
            node_id = start
        else:
            node_id = serial_number.next()
        node = ConstituencyTreeNode(labels, text, node_id=node_id, start=start, end=end, depth=depth)
        children = list(sentence._.children)
        for child in children:
            node.children.append(ConstituencyTreeNode._inner_parse_using_spacy(child, serial_number, depth=depth+1))
        return node


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(BeneparComponent('benepar_en2'))
    sentence = 'Great taste bad service.'
    tree = ConstituencyTreeNode.parse_using_spacy(nlp, sentence)
    # adjacency_list = tree.get_adjacency_list()
    all_nodes = ConstituencyTreeNode.get_all_nodes(tree)
    adjacency_list = tree.get_adjacency_list_between_all_node_and_leaf()
    edges = [['%s-%s' % (e2.node_id, e2.text) for e2 in e1] for e1 in adjacency_list]
    g = nx.DiGraph()
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos=pos, with_labels=True)
    plt.show()
    print()