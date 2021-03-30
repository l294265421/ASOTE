# -*- coding: utf-8 -*-


from typing import Dict, List
import logging
import json
import requests


class PosConvert(object):

    def __call__(self,
                 text: str,
                 pos: int,
                 encoding:str,
                 *args,
                 **kwargs)->int:
        """
        位置转换
        :param text: 文本
        :param pos: 当文本是from_encoding时候的位置
        :param encoding: 原始encoding
        :param args:
        :param kwargs:
        :return: 当前text中实际的位置
        """
        encode_text = text.encode(encoding)

        segment = bytes(encode_text[0: pos])

        return len(segment.decode(encoding))


class NerPosRemote(object):
    """
    命名实体识别 remote
    """

    _URL = ""

    def _request(self, post_data: Dict):

        post_data_str = json.dumps(post_data, ensure_ascii=False)
        r = requests.post(NerPosRemote._URL,
                          post_data_str.encode(encoding='utf-8'),
                          timeout=15)

        r.encoding = 'utf-8'

        if r.status_code == requests.codes.ok:
            try:
                results = json.loads(r.text)

                pos_convert = PosConvert()
                for result in results.get("results", list()):

                    result_text = result["text"]

                    for item in result.get("items", list()):
                        ne_text = item["item"]
                        byte_offset = item["byte_offset"]

                        begin = pos_convert(text=result_text,
                                            pos=byte_offset,
                                            encoding="gb18030")

                        item["byte_length"] = len(ne_text)
                        item["byte_offset"] = begin

            except Exception as e:
                logging.fatal("error: " + str(e))
            return results
        else:
            logging.fatal('none result')

    def __call__(self, text: [str, List], *args, **kwargs):
        ner_result = None

        if not isinstance(text, list):
            text = [text]

        post_data = {"texts": text}

        try:
            ner_result = self._request(post_data)

        except Exception as e:
            logging.info('none result: %s' % str(e))
        return ner_result


if __name__ == '__main__':
    text = '华为一直说自己5G强运行快，但媒体实测连5G速度都不如三星'
    ner_pos = NerPosRemote()
    result = ner_pos(text)
    print(result)
