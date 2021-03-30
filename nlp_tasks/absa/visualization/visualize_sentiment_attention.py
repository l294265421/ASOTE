import os
import warnings
import re

import numpy as np
from keras.preprocessing import text, sequence

from nlp_tasks.absa.utils import data_utils
from nlp_tasks.absa.models import keras_models
from nlp_tasks.absa.utils import embedding_utils, result_utils, evaluate_utils
from nlp_tasks.absa.utils import file_utils, tokenizer_utils
from nlp_tasks.absa.conf import model_path, data_path, thresholds

np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '8'

train_file_path = data_path.train_sentiment_value_word_exact_file_path
test_file_path = data_path.test_public_for_sentiment_value_word_exact_file_path

x_train = data_utils.read_features(train_file_path)
# y_train = data_utils.read_labels(train_file_path)
# train_subjects = data_utils.read_subject_of_sentiment_value(train_file_path)
# y_train = np.array(y_train)

x_test = data_utils.read_features(test_file_path)
# test_subjects = data_utils.read_subject_of_sentiment_value(test_file_path)
# id_test = data_utils.read_ids(test_file_path)

max_features = 30000
embed_size = 100

tokenizer = tokenizer_utils.get_tokenizer(topic_or_sentiment='sentiment')

x_train = tokenizer.texts_to_sequences(x_train)
# x_train_len = [len(element) for element in x_train]
# train_subjects_repeated = data_utils.repeat_element_in_list(train_subjects, x_train_len)
# x_train_subject = tokenizer.texts_to_sequences(train_subjects_repeated)

x_test = tokenizer.texts_to_sequences(x_test)
# x_test_len = [len(element) for element in x_test]
# test_subjects_repeated = data_utils.repeat_element_in_list(test_subjects, x_test_len)
# x_test_subject = tokenizer.texts_to_sequences(test_subjects_repeated)

maxlen = data_utils.max_len(x_train + x_test)

# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_train_subject = sequence.pad_sequences(x_train_subject, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# x_test_subject = sequence.pad_sequences(x_test_subject, maxlen=maxlen)

word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index) + 1)
embedding_matrix = embedding_utils.generate_embedding_matrix(word_index, nb_words,
                                                             data_path.embedding_file,
                                                             embed_size)

model_name = 'rnn_attention_sentiment'
model_filepath = model_path.model_file_dir + model_name + '.hdf5'
model = keras_models.at_lstm(maxlen, nb_words, embed_size, embedding_matrix, 3)
model.load_weights(model_filepath)


def preict(data, model):
    for i in range(len(data)):
        sample = data[i]
        parts = sample.split(',')
        text = parts[1]
        x_train = tokenizer.texts_to_sequences([text])
        x_train_len = [len(element) for element in x_train]
        subject = parts[2]
        subject_repeated = data_utils.repeat_element_in_list([subject], x_train_len)
        x_subject = tokenizer.texts_to_sequences(subject_repeated)

        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_subject = sequence.pad_sequences(x_subject, maxlen=maxlen)

        prob = model.predict([x_train, x_subject])
        predict_label = result_utils.convert_sentiment_value_predict(prob)
        print(predict_label)

train_data = file_utils.read_all_lines(train_file_path)
test_data = file_utils.read_all_lines(test_file_path)

print('train')
train_sample = preict(train_data, model)

print('test')
test_sample = preict(test_data, model)
