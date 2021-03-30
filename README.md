# The code and data for the paper "A More Fine-Grained Aspect-Sentiment-Opinion Triplet Extraction Task" [paper](https://arxiv.org/pdf/2103.15255.pdf)

# Differences between ASOTE and ASTE
![](figures/asote_vs_aste.png)
In the third sentence, the negative sentiment toward the aspect term “food” isexpressed without an annotatable opinion.

# Requirements
- Python 3.6.8
- torch==1.2.0
- pytorch-transformers==1.1.0
- allennlp==0.9.0

# Instructions:
1. Before excuting the following commands, replace glove.840B.300d.txt(http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip), bert-base-uncased.tar.gz(https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and vocab.txt(https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) with the corresponding absolute paths in your computer. 
2. We only provide the 14res dataset for review.

# ATE
scripts/ate.sh

# TOWE
scripts/towe.sh

# TOWE inference
scripts/towe.predic.sh

# AOSC
scripts/tosc.sh

# U-ASO
scripts/towe_tosc_jointly.sh

# U-ASO inference
scripts/towe_tosc_jointly.predict.sh

# MIL-ASO
scripts/mil_aso.sh

# MIL-ASO inference
scripts/mil_aso.predict.sh

# evaluate
scripts/evaluate.sh