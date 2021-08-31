import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

original_data_dir = os.path.join(project_dir, 'ASOTE-data')

common_data_dir = project_dir + '/model_data/'

common_code_dir = project_dir + '/nlp_tasks/'

stopwords_filepath = original_data_dir + 'common/stopwords.txt'


def get_task_data_dir(task_name: str, is_original=False):
    """

    :param task_name:
    :return:
    """
    if not is_original:
        return os.path.join(common_data_dir, task_name) + '/'
    else:
        return os.path.join(original_data_dir, task_name) + '/'
