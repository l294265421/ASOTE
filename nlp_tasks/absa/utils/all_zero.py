from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

train_data_lines = file_utils.read_all_lines(data_path.train_file_path)[1:]
train_data_content_line_map = {}
for train_data_line in train_data_lines:
    parts = train_data_line.split(',')
    if parts[1] not in train_data_content_line_map:
        train_data_content_line_map[parts[1]] = []
    train_data_content_line_map[parts[1]].append(train_data_line)

test_public_for_sentiment_lines = file_utils.read_all_lines(data_path.test_public_for_sentiment_value_file_path)
result = ['content_id,subject,sentiment_value,sentiment_word']
in_train_data = set()
in_train_data_for_submit = []
for test_public_for_sentiment_line in test_public_for_sentiment_lines:
    parts = test_public_for_sentiment_line.split(',')
    if parts[1] in train_data_content_line_map:
        if parts[1] not in in_train_data:
            in_train_data.add(parts[1])
            in_train_data_samples = train_data_content_line_map[parts[1]]
            for in_train_data_sample in in_train_data_samples:
                in_train_data_sample_parts = in_train_data_sample.split(',')
                result.append(parts[0] + ',' + in_train_data_sample_parts[2] + ',' + in_train_data_sample_parts[3] + ',')
                in_train_data_for_submit.append(parts[0] + ',' + in_train_data_sample_parts[2] + ',' + in_train_data_sample_parts[3] + ',')
    else:
        result.append(parts[0] + ',' + parts[2] + ',0,')
file_utils.write_lines(result, data_path.data_base_dir + 'all_zero.result')
file_utils.write_lines(in_train_data_for_submit, data_path.data_base_dir + 'in_train_data_for_submit')
