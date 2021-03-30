from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import file_utils

in_train_data_for_submit = file_utils.read_all_lines(data_path.data_base_dir + 'in_train_data_for_submit')
in_train_data_for_submit_id = [line.split(',')[0] for line in in_train_data_for_submit]

result_file_name = 'test_public.result_20181028232554_caokong_xingneng.csv'
result = file_utils.read_all_lines(data_path.data_base_dir + result_file_name)
merge_result = [result.pop(0)]
for line in result:
    id = line.split(',')[0]
    if id in in_train_data_for_submit_id:
        continue
    else:
        merge_result.append(line)

merge_result.extend(in_train_data_for_submit)
file_utils.write_lines(merge_result, data_path.data_base_dir + result_file_name + '.merge_result_and_in_train')