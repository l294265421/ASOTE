# -*- coding: utf-8 -*-


import logging
import re
import traceback

from nlp_tasks.utils.my_corenlp import StanfordCoreNLP
from nlp_tasks.utils import corenlp_factory
from nlp_tasks.common import common_path

MODEL_DIR = common_path.common_data_dir + 'stanford-corenlp-full-2018-02-27/'


class TreeNode:
    """
    成分句法树节点
    """
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @staticmethod
    def to_string(node, recursive=False, filter_node: list=None):
        """

        :param node:
        :param recursive:
        :param filter_pos: 词性在filter_pos里的词及其后代不会用于生成字符串
        :return:
        """
        if node is None:
            return ''
        if recursive:
            leaves = TreeNode.get_all_leaves(node, filter_node=filter_node)
            values_of_leaves = [leaf.value for leaf in leaves]
            return ''.join(values_of_leaves)

    @staticmethod
    def get_all_leaves(root, filter_node: list=None):
        """
        获得所有叶子节点
        :param root:
        :param filter_node: 节点名出现在filter_node里的节点的后代叶子节点不会返回
        :return:
        """
        if root is None:
            return []
        if filter_node is None:
            filter_node = []
        if len(root.children) == 0:
            return [root]
        else:
            result = []
            for child in root.children:
                if child.value not in filter_node:
                    result.extend(TreeNode.get_all_leaves(child, filter_node=filter_node))
            return result

    @staticmethod
    def get_np_ancestor(node, root):
        """
        获得包含node的最上层NP节点，但不包含根节点
        :param node:
        :param root:
        :return: 没有找到返回None
        """
        result = None
        if node is None:
            return result
        cursor = node
        while cursor.parent is not None and cursor.parent != root:
            cursor = cursor.parent
            if cursor.value == 'NP':
                result = cursor
        return result

    @staticmethod
    def get_ancestor(node, root, target_ancestor_value, is_top=False):
        """
        获得包含node的值为target_ancestor_value的节点，但不包含根节点
        :param node: 当前节点
        :param root: 根节点
        :param target_ancestor_value: 要找的祖先节点的值
        :param is_top: true时找最上层的祖先，否则找最近的祖先
        :return:
        """
        result = None
        if node is None or root is None or target_ancestor_value is None:
            return result
        cursor = node
        while cursor.parent is not None and cursor.parent != root:
            cursor = cursor.parent
            if cursor.value == target_ancestor_value:
                result = cursor
                if not is_top:
                    break
        return result

    @staticmethod
    def is_sub_tree(tree_candidate, sub_tree_candidate):
        """
        判断sub_tree_candidate是否是tree_candidate的子树
        :param tree_candidate:
        :param sub_tree:
        :return:
        """
        if tree_candidate is None or sub_tree_candidate is None:
            return False
        # sub_tree_candidate与tree_candidate的每个节点比较
        result = tree_candidate == sub_tree_candidate
        if result:
            return result
        else:
            for child in tree_candidate.children:
                result = result or TreeNode.is_sub_tree(child, sub_tree_candidate)
            return result

    @staticmethod
    def find_corresponding_node(root, value, original_index):
        """
        从成分解析树中找到值为value的相应节点
        :param root:
        :param value:
        :param original_index:
        :return:
        """
        leaves = TreeNode.get_all_leaves(root)
        if len(leaves) == 0:
            return None
        try:
            result = leaves[original_index]
        except:
            logging.error('%s-%s' % (traceback.format_exc(), str(original_index)))
            result = None
        return result


def sub_constituency_parser_result_generator(constituency_parser_result: str):
    """
    输入：(NR 中) (NR 美)
    输出：
    (NR 中)
    (NR 美)
    :param constituency_parser_result:
    :return:
    """
    result = []
    if not constituency_parser_result.startswith('('):
        result.append(constituency_parser_result)
        return result

    stack = []
    start_index = -1
    for i, c in enumerate(constituency_parser_result):
        if c == '(':
            stack.append('(')
            if start_index == -1:
                start_index = i
        elif c == ')':
            stack.pop()
            if len(stack) == 0:
                result.append(constituency_parser_result[start_index: i + 1])
                start_index = -1
    return result


def parse_corenlp_parse_result(constituency_parser_result: str):
    """
    中美军事基地相邻导致紧张局势不断持续。
    输入：
    (ROOT
      (IP
        (NP
          (NP (NR 中) (NR 美))
          (NP (NN 军事) (NN 基地)))
        (VP (VV 相邻)
          (IP
            (VP (VV 导致)
              (IP
                (NP
                  (ADJP (JJ 紧张))
                  (NP (NN 局势)))
                (VP
                  (ADVP (AD 不断))
                  (VP (VV 持续)))))))
        (PU 。)))
    输出：一棵树
    观察：
    1. 从根节点开始解析
    2. 除叶子节点外，节点这样开始 (${节点名}；叶子节点只包含文字
    3. 每个节点这样结束 )
    方案1. 递归
    :param constituency_parser_result:
    :return:
    """
    if constituency_parser_result is None:
        return None
    constituency_parser_result = re.sub('\\s+', ' ', constituency_parser_result)
    if not constituency_parser_result.startswith('('):
        leaf_node = TreeNode(constituency_parser_result)
        return leaf_node
    else:
        constituency_parser_result = constituency_parser_result[1: -1]
        first_whitespace_index = constituency_parser_result.index(' ')
        value = constituency_parser_result[: first_whitespace_index]
        parent_node = TreeNode(value)
        constituency_parser_result = constituency_parser_result[first_whitespace_index + 1:]
        sub_constituency_parser_results = sub_constituency_parser_result_generator(constituency_parser_result)
        children = []
        for sub_constituency_parser_result in sub_constituency_parser_results:
            parse_result = parse_corenlp_parse_result(sub_constituency_parser_result)
            children.append(parse_result)
        for child in children:
            child.parent = parent_node
        parent_node.children = children
        return parent_node


if __name__ == '__main__':
    with corenlp_factory.create_corenlp_server(lang='zh', start_new_server=True) as nlp:
        sentence = '【多国隐形战机“暗战”巴黎航展】图片说明：航空工业展台的隐形战机模型引发关注。'
        sentence = re.sub('[\(\)]', '', sentence)
        result = nlp.parse(sentence)
        print(result)
        constituency_tree = parse_corenlp_parse_result(result)
        print(TreeNode.find_corresponding_node(constituency_tree, '模型', 20))
        print('end')

