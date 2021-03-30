"""

"""
import argparse
import sys
import random
import copy

import torch
import numpy

from nlp_tasks.absa.utils import argument_utils
from nlp_tasks.absa.mining_opinions.tosc import tosc_train_templates as templates


parser = argparse.ArgumentParser()
parser.add_argument('--current_dataset', help='dataset name', default='RealASOTripletRest16', type=str)
parser.add_argument('--task_name', help='task name', default='tosc', type=str)
parser.add_argument('--data_type', help='task name', default='common', type=str)
parser.add_argument('--model_name', help='model name', default='LSTM', type=str)
parser.add_argument('--timestamp', help='timestamp', default=int(1571400646), type=int)
parser.add_argument('--train', help='if train a new model', default=False, type=argument_utils.my_bool)
parser.add_argument('--evaluate', help='evaluate', default=False, type=argument_utils.my_bool)
parser.add_argument('--predict', help='predict given samples', default=False, type=argument_utils.my_bool)
parser.add_argument('--predict_test', help='predict test set', default=True, type=argument_utils.my_bool)
parser.add_argument('--epochs', help='epochs', default=100, type=int)
parser.add_argument('--batch_size', help='batch_size', default=32, type=int)
parser.add_argument('--patience', help='patience', default=10, type=int)
parser.add_argument('--visualize_attention', help='visualize attention', default=False, type=argument_utils.my_bool)
parser.add_argument('--embedding_filepath', help='embedding filepath',
                    default='D:\program\word-vector\glove.840B.300d.txt', type=str)
parser.add_argument('--embed_size', help='embedding dim', default=300, type=int)
parser.add_argument('--seed', default=776, type=int)
parser.add_argument('--repeat', default='0', type=str)
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--layer_number_of_lstm', default=3, type=int)
parser.add_argument('--position', default=False, type=argument_utils.my_bool)
parser.add_argument('--position_embeddings_dim', help='position embeddings dim', default=32, type=int)
parser.add_argument('--debug', default=False, type=argument_utils.my_bool)
parser.add_argument('--early_stopping_by_batch', default=False, type=argument_utils.my_bool)


parser.add_argument('--data_augmentation', default=False, type=argument_utils.my_bool)

parser.add_argument('--max_len', help='max length', default=100, type=int)
parser.add_argument('--bert_file_path', help='bert_file_path',
                    default=r'D:\program\word-vector\bert-base-uncased.tar.gz', type=str)
parser.add_argument('--bert_vocab_file_path', help='bert_vocab_file_path',
                    default=r'D:\program\word-vector\uncased_L-12_H-768_A-12\vocab.txt', type=str)
parser.add_argument('--fixed_bert', default=False, type=argument_utils.my_bool)
parser.add_argument('--learning_rate_in_bert', default=2e-5, type=float)
parser.add_argument('--l2_in_bert', default=0.00001, type=float)

parser.add_argument('--gat_visualization', help='gat visualization', default=False, type=argument_utils.my_bool)

parser.add_argument('--sample_mode', help='sample mode', default='multi', type=str,
                    choices=['single', 'multi'])
# parser.add_argument('--all', help='predict the sentiments of all aspect terms appearing in sentences', default=False, type=argument_utils.my_bool)
parser.add_argument('--aspect_term_aware', help='insert a special token at both the beginning and end of aspect terms', default=False, type=argument_utils.my_bool)
parser.add_argument('--term', help='predict the sentiment of the aspect term based on the representation of the aspect term', default=False, type=argument_utils.my_bool)
parser.add_argument('--cls', help='predict the sentiment of the aspect term based on the representations of both the cls and the aspect term', default=False, type=argument_utils.my_bool)
parser.add_argument('--pair', help='the aspect term as the second sentence of BERT\'s input', default=False, type=argument_utils.my_bool)
parser.add_argument('--syntax', help='apply a gnn layer on the output of BERT to capture the syntax information', default=False, type=argument_utils.my_bool)
parser.add_argument('--mean_or_cat_of_term_and_cls', help='mean_or_cat_of_term_and_cls', default='mean', type=str,
                    choices=['mean', 'cat'])

parser.add_argument('--same_special_token', default=False, type=argument_utils.my_bool)

parser.add_argument('--ate_result_filepath', help='ate result filepath',
                    default='', type=str)

parser.add_argument('--consider_target', default=True, type=argument_utils.my_bool)

parser.add_argument('--second_sentence', default=False, type=argument_utils.my_bool)

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)
gpu_ids = args.gpu_id.split(',')
if len(gpu_ids) == 1:
    args.gpu_id = -1 if int(gpu_ids[0]) == -1 else 0
else:
    args.gpu_id = list(range(len(gpu_ids)))


configuration = args.__dict__

if configuration['seed'] is not None:
    random.seed(configuration['seed'])
    numpy.random.seed(configuration['seed'])
    torch.manual_seed(configuration['seed'])
    torch.cuda.manual_seed(configuration['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# data_type must contatin the options that affect the data file
# configuration['data_type'] = 'data_type.{data_type}-sample_mode.{sample_mode}-aspect_term_aware.{aspect_term_aware}-pair.{pair}-consider_target.{consider_target}'.format_map(configuration)

model_name_complete_prefix = 'model_name_{model_name}'.format_map(configuration)

configuration_for_this_repeat = copy.deepcopy(configuration)
configuration_for_this_repeat['model_name_complete'] = '%s.%s' % (model_name_complete_prefix, args.repeat)

model_name = configuration['model_name']
if model_name in ['LSTM']:
    template = templates.SpanBasedModelForAtsa(configuration_for_this_repeat)
if model_name in ['bert']:
    template = templates.SpanBasedBertModelForTOSC(configuration_for_this_repeat)
# elif model_name in ['Bert', 'aspect-term-aware-Bert']:
#     template = templates.SpanBasedBertModelForAtsa(configuration_for_this_repeat)
# elif model_name in ['Bert-syntax', 'aspect-term-aware-bert-syntax']:
#     template = templates.SyntaxAwareSpanBasedBertModelForAtsa(configuration_for_this_repeat)
# elif model_name in ['AtsaBERT']:
#     template = templates.AtsaBERT(configuration_for_this_repeat)
# elif model_name in ['AtsaLSTM']:
#     template = templates.AtsaLSTM(configuration_for_this_repeat)
elif model_name in ['']:
    pass

if configuration_for_this_repeat['train']:
    template.train()
if configuration_for_this_repeat['evaluate']:
    template.evaluate()
if configuration_for_this_repeat['predict_test']:
    output_filepath = template.model_dir + 'result_of_predicting_test.txt'
    print('result_of_predicting_test:%s ' % output_filepath)
    template.predict_test(output_filepath)
if configuration_for_this_repeat['predict']:
    result = template.predict()
print()
