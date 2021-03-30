# -*- coding: utf-8 -*-


import json
import logging
import traceback
import copy

from nlp_tasks.utils import http_utils


def search(es_url, post_data, max_hits_needed=10, size=10):
    """
    查询es
    :param es_url: es地址
    :param post_data: 查询请求post中的数据
    :param max_hits_needed: 需要的最大数据量
    :return: es查询结果的hits部分
    """
    url = es_url
    post_data_copy = copy.deepcopy(post_data)
    es_from = 0
    result = []
    while True:
        try:
            post_data_copy['from'] = es_from
            post_data_copy['size'] = size
            es_result = http_utils.post(url, post_data_copy)
            es_result_obj = json.loads(es_result)
            hits = es_result_obj['hits']['hits']
            result.extend(hits)
            if len(hits) < size or len(result) > max_hits_needed:
                break
            else:
                es_from += size
        except Exception as e:
            logging.error('查询es失败: %s' % traceback.format_exc())
            break
    return result[: max_hits_needed]


def search_by_scroll(es_url, post_data, index):
    """
    查询es
    :param es_url: es地址
    :param post_data: 查询请求post中的数据
    :param index: es的索引
    :return: es查询结果的hits部分
    """
    end_index = es_url.index(':', 5)
    es_url = es_url[: end_index + 5]
    size = 10000
    post_data_copy = copy.deepcopy(post_data)
    result = []
    scroll_id = ''
    while True:
        try:
            if scroll_id:
                params = {}
                params['scroll'] = '5m'
                params['scroll_id'] = scroll_id
                url = '%s/_search/scroll?' % es_url
                es_result = http_utils.get(url, params)
            else:
                url = es_url + ('/%s' % index) + '/_search?scroll=1m&size=' + str(size)
                es_result = http_utils.post(url, post_data_copy)
            es_result_obj = json.loads(es_result)
            hits = es_result_obj['hits']['hits']
            scroll_id = es_result_obj['_scroll_id']
            result.extend(hits)
            if len(hits) < size:
                break
        except Exception as e:
            logging.error('查询es失败: %s' % traceback.format_exc())
            break
    return result


def search_esproxy_by_scroll(es_url, scroll_url_template, post_data, size):
    """
    查询es
    :param es_url: es地址
    :param scroll_url_template: scroll时的地址
    :param post_data: 查询请求post中的数据
    :return: es查询结果的hits部分
    """
    post_data_copy = copy.deepcopy(post_data)
    result = []
    scroll_id = ''
    while True:
        try:
            if scroll_id:
                params = {}
                scroll_url = scroll_url_template % scroll_id
                es_result = http_utils.get(scroll_url, params)
            else:
                es_result = http_utils.post(es_url, post_data_copy)
            es_result_obj = json.loads(es_result)
            hits = es_result_obj['hits']['hits']
            scroll_id = es_result_obj['_scroll_id']
            result.extend(hits)
            if len(hits) < size:
                break
        except Exception as e:
            logging.error('查询es失败: %s' % traceback.format_exc())
            break
    return result
