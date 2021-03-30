# -*- coding: utf-8 -*-
"""

Date:    2018/10/12 15:32
"""

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.utils import file_utils

if __name__ == '__main__':
    result = []
    topics = label_mapping.subject_mapping.keys()
    for topic in topics:
        result += file_utils.\
            read_all_lines(data_path.test_public_sentiment_value_result_file_path + '.' + topic)
    file_utils.write_lines(result, data_path.test_public_sentiment_value_result_file_path)