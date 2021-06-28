# 36
sh repeat_non_bert.sh 3 36-ASOTEDataRest14-0,36-ASOTEDataRest14-1,36-ASOTEDataRest14-2,36-ASOTEDataRest14-3,36-ASOTEDataRest14-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --current_dataset ASOTEDataRest14 --data_type common_unified_tag --model_name AsoTermModel--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.36-ASOTEDataRest14-0.log 2>&1 &

sh repeat_non_bert.sh 3 36-ASOTEDataLapt14-0,36-ASOTEDataLapt14-1,36-ASOTEDataLapt14-2,36-ASOTEDataLapt14-3,36-ASOTEDataLapt14-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --current_dataset ASOTEDataLapt14 --data_type common_unified_tag --model_name AsoTermModel--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.36-ASOTEDataLapt14-0.log 2>&1 &

sh repeat_non_bert.sh 3 36-ASOTEDataRest15-0,36-ASOTEDataRest15-1,36-ASOTEDataRest15-2,36-ASOTEDataRest15-3,36-ASOTEDataRest15-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --current_dataset ASOTEDataRest15 --data_type common_unified_tag --model_name AsoTermModel--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.36-ASOTEDataRest15-0.log 2>&1 &

sh repeat_non_bert.sh 3 36-ASOTEDataRest16-0,36-ASOTEDataRest16-1,36-ASOTEDataRest16-2,36-ASOTEDataRest16-3,36-ASOTEDataRest16-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --current_dataset ASOTEDataRest16 --data_type common_unified_tag --model_name AsoTermModel--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.36-ASOTEDataRest16-0.log 2>&1 &

# 4
sh repeat_non_bert.sh 0 4-ASOTEDataRest14-0,4-ASOTEDataRest14-1,4-ASOTEDataRest14-2,4-ASOTEDataRest14-3,4-ASOTEDataRest14-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataRest14 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.4-ASOTEDataRest14-0.log 2>&1 &

sh repeat_non_bert.sh 0 4-ASOTEDataLapt14-0,4-ASOTEDataLapt14-1,4-ASOTEDataLapt14-2,4-ASOTEDataLapt14-3,4-ASOTEDataLapt14-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataLapt14 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.4-ASOTEDataLapt14-0.log 2>&1 &

sh repeat_non_bert.sh 2 4-ASOTEDataRest15-0,4-ASOTEDataRest15-1,4-ASOTEDataRest15-2,4-ASOTEDataRest15-3,4-ASOTEDataRest15-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataRest15 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.4-ASOTEDataRest15-0.log 2>&1 &

sh repeat_non_bert.sh 2 4-ASOTEDataRest16-0,4-ASOTEDataRest16-1,4-ASOTEDataRest16-2,4-ASOTEDataRest16-3,4-ASOTEDataRest16-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataRest16 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 > aso_unified_tag.4-ASOTEDataRest16-0.log 2>&1 &

# 10
sh repeat_non_bert.sh 3 10-ASOTEDataRest14-0,10-ASOTEDataRest14-1,10-ASOTEDataRest14-2,10-ASOTEDataRest14-3,10-ASOTEDataRest14-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataRest14 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 --fixed_bert False > aso_unified_tag.10-ASOTEDataRest14-0.log 2>&1 &

sh repeat_non_bert.sh 3 10-ASOTEDataLapt14-0,10-ASOTEDataLapt14-1,10-ASOTEDataLapt14-2,10-ASOTEDataLapt14-3,10-ASOTEDataLapt14-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataLapt14 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 --fixed_bert False > aso_unified_tag.10-ASOTEDataLapt14-0.log 2>&1 &

sh repeat_non_bert.sh 3 10-ASOTEDataRest15-0,10-ASOTEDataRest15-1,10-ASOTEDataRest15-2,10-ASOTEDataRest15-3,10-ASOTEDataRest15-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataRest15 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 --fixed_bert False > aso_unified_tag.10-ASOTEDataRest15-0.log 2>&1 &

sh repeat_non_bert.sh 3 10-ASOTEDataRest16-0,10-ASOTEDataRest16-1,10-ASOTEDataRest16-2,10-ASOTEDataRest16-3,10-ASOTEDataRest16-4 nlp_tasks/absa/mining_opinions/sequence_labeling/aso_bootstrap.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path uncased_L-12_H-768_A-12/vocab.txt --current_dataset ASOTEDataRest16 --data_type common_unified_tag_bert --model_name AsoTermModelBert--train True --evaluate True --predict False --crf False --validation_metric +opinion_sentiment_f1 --fixed_bert False > aso_unified_tag.10-ASOTEDataRest16-0.log 2>&1 &
