# -*- coding: utf-8 -*-


from nlp_tasks.utils import stanfordnlp_sentence_constituency_parser as constip


def get_event_representation_by_constituency_tree(target_node: constip.TreeNode, constituency_tree: constip.TreeNode):
    """
    1. 返回的节点不会是根节点(constituency_tree)
    :param target_node:
    :param original_index:
    :param constituency_tree:
    :return:
    """
    if target_node is None:
        return None

    # if successful_rule == 'rule5' and cause_or_effect == 'effect':
    #     result = constip.TreeNode.get_ancestor(target_node, constituency_tree, 'IP')
    #     return result

    # 如果是名词，抽取包含核心词的名词短语
    if target_node.parent is not None and target_node.parent.value.startswith('N'):
        result = constip.TreeNode.get_np_ancestor(target_node, constituency_tree)
        return result if result != constituency_tree else None

    # 核心词 <– VV <- VP -> NP -> NN -> 名词
    # 对应case: 赶往现场指挥救援，
    if target_node.parent is not None and target_node.parent.value == 'VV' \
            and target_node.parent.parent is not None and target_node.parent.parent.value == 'VP' \
            and len(target_node.parent.parent.children) == 2 \
            and 'NP' in [child.value for child in target_node.parent.parent.children]:
        result = target_node.parent.parent
        return result if result != constituency_tree else None

    # 如果是动词，且在成分分析树中存在这条路径，核心词 <– VV <- VP <- (IP or VP)，就把IP下的叶结点作为事件表示
    if target_node.parent is not None and target_node.parent.value.startswith('V') \
        and target_node.parent.parent is not None and target_node.parent.parent.value == 'VP' \
        and target_node.parent.parent.parent is not None and target_node.parent.parent.parent.value in ('IP', 'VP'):
        result = target_node.parent.parent.parent
        return result if result != constituency_tree else None
    return None


def get_ccomp_event_representation_by_constituency_tree(target_node: constip.TreeNode, constituency_tree: constip.TreeNode):
    """
    1. 返回的节点不会是根节点(constituency_tree)
    :param target_node:
    :param original_index:
    :param constituency_tree:
    :return:
    """
    if target_node is None:
        return None

    result = constip.TreeNode.get_ancestor(target_node, constituency_tree, 'IP')
    return result


def get_noun_event_representation_by_constituency_tree(target_node: constip.TreeNode,
                                                       constituency_tree: constip.TreeNode):
    """
    1. 返回的节点不会是根节点(constituency_tree)
    :param target_node:
    :param original_index:
    :param constituency_tree:
    :return:
    """
    if target_node is None:
        return None

    # 如果是名词，抽取包含核心词的名词短语
    if target_node.parent is not None and target_node.parent.value.startswith('N'):
        result = constip.TreeNode.get_np_ancestor(target_node, constituency_tree)
        return result if result != constituency_tree else None
    return None


def get_verb_event_representation_by_constituency_tree(target_node: constip.TreeNode,
                                                       constituency_tree: constip.TreeNode):
    """
    1. 返回的节点不会是根节点(constituency_tree)
    :param target_node:
    :param original_index:
    :param constituency_tree:
    :return:
    """
    if target_node is None:
        return None

    # 如果是动词，且在成分分析树中存在这条路径，核心词 <– VV <- VP <- IP，就把IP下的叶结点作为事件表示
    if target_node.parent is not None and target_node.parent.value.startswith('V') \
        and target_node.parent.parent is not None and target_node.parent.parent.value == 'VP' \
        and target_node.parent.parent.parent is not None and target_node.parent.parent.parent.value == 'IP':
        result = target_node.parent.parent.parent
        return result if result != constituency_tree else None
    return None
