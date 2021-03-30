import pandas as pd

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.utils import result_utils

if __name__ == '__main__':
    sentiment_value_probability_file_path = \
        ['test_public_sentiment_value_probability.result.rnn_attention_cv_sentiment',
         'test_public_sentiment_value_probability.result.interactive_attention_cv_sentiment',
         'test_public_sentiment_value_probability.result.rnn_attention_random_cv_sentiment']
    sentiments = []
    for file_path in sentiment_value_probability_file_path:
        df = pd.read_csv(data_path.data_base_dir + file_path)
        id = df['id']
        id_list = id.tolist()
        sentiment = df[['-1', '0', '1']]
        sentiment_values = sentiment.values
        sentiments.append(sentiment_values)
    y_test = sentiments[0]
    for i in range(1, len(sentiments)):
        y_test += sentiments[i]
    y_test /= len(sentiments)
    result_utils.save_sentiment_value_result(y_test, id, 'sentiment_average')