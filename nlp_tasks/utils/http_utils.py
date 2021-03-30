# -*- coding: utf-8 -*-


import json
import urllib.request as request
import requests


def post(url, data):
    """
    发送post请求
    :param url: str, url地址
    :param data: dict, post请求的查询数据
    :return: str, 请求返回的数据
    """
    data = bytes(json.dumps(data), encoding="utf-8")
    request_obj = request.Request(url, headers={'Content-Type': 'application/json'})

    with request.urlopen(request_obj, data=data) as f:
        result = f.read()
        result = str(result, encoding='utf-8')
    return result


def get(url, params):
    """
    发送post请求
    :param url: str, url地址
    :param data: dict, 参数
    :return: str, 请求返回的数据
    """
    response = requests.get(url=url, headers={'Content-Type': 'application/json'}, params=params)
    return response.text


if __name__ == '__main__':
    url = 'http://10.194.6.49/ESProxyService/queryTable?user=qaMonitor&logid=qaxiaojuan' \
          '&size=100&table=real_tieba_20190129,real_tieba_20190130,real_tieba_20190131'
    # post_data = {"query": {"term": {'thread_id': '6020285360'}}}
    should = []
    for word in ['白粉']:
        should.append({'match_phrase': {"content": {"query": word}}})
    post_data = {"query": {"bool": {"should": should}}}
    print(post(url, post_data))
