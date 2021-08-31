class NerBertForOTE(NerLstmForOTE):
    """

    """

    def __init__(self, configuration):
        self.bert_file_path = configuration['bert_file_path']
        self.bert_vocab_file_path = configuration['bert_vocab_file_path']
        self.max_len = configuration['max_len']
        super().__init__(configuration)

    def _get_data_reader(self):
        token_indexer = SingleIdTokenIndexer(namespace="tokens")
        position_indexer = SingleIdTokenIndexer(namespace='position')
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file_path, do_lower_case=True)
        bert_token_indexer = WordpieceIndexer(vocab=bert_tokenizer.vocab,
                                              wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                                              namespace="bert",
                                              use_starting_offsets=False,
                                              max_pieces=self.max_len,
                                              do_lowercase=True,
                                              never_lowercase=None,
                                              start_tokens=None,
                                              end_tokens=None,
                                              separator_token="[SEP]",
                                              truncate_long_sequences=True)
        reader = sequence_labeling_data_reader.DatasetReaderForNerBertForOTEOfRealASO(
            tokenizer=self._get_word_segmenter(),
            token_indexers={"tokens": token_indexer},
            position_indexers={'position': position_indexer},
            configuration=self.configuration,
            bert_tokenizer=bert_tokenizer,
            bert_token_indexers={"bert": bert_token_indexer}
        )
        return reader

    def _get_bert_word_embedder(self):
        # bert_embedder = PretrainedBertEmbedder(
        #     pretrained_model=self.bert_file_path,
        #     top_layer_only=True,  # conserve memory
        #     requires_grad=(not self.configuration['fixed'])
        # )

        pretrained_model = self.bert_file_path
        bert_model = PretrainedBertModel.load(pretrained_model, cache_model=False)
        for param in bert_model.parameters():
            param.requires_grad = (not self.configuration['fixed_bert'])
        bert_embedder = BertEmbedder(bert_model=bert_model, top_layer_only=True)

        bert_word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"bert": bert_embedder},
                                                                       # we'll be ignoring masks so we'll need to set this to True
                                                                       allow_unmatched_keys=True)
        bert_word_embedder.to(self.configuration['device'])
        return bert_word_embedder

    def _find_model_function_pure(self):
        return pytorch_models.NerBert

    def _find_model_function(self):
        embedding_dim = self.configuration['embed_size']
        embedding_matrix_filepath = self.base_data_dir + 'embedding_matrix'
        if os.path.exists(embedding_matrix_filepath):
            embedding_matrix = super()._load_object(embedding_matrix_filepath)
        else:
            embedding_filepath = self.configuration['embedding_filepath']
            embedding_matrix = embedding._read_embeddings_from_text_file(embedding_filepath, embedding_dim,
                                                                         self.vocab, namespace='tokens')
            super()._save_object(embedding_matrix_filepath, embedding_matrix)
        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='tokens'),
                                    embedding_dim=embedding_dim, padding_index=0, vocab_namespace='tokens',
                                    trainable=False, weight=embedding_matrix)
        # the embedder maps the input tokens to the appropriate embedding matrix
        word_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        position_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size(namespace='position'),
                                    embedding_dim=self._get_position_embeddings_dim(), padding_index=0)
        position_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({"position": position_embedding},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)

        bert_word_embedder = self._get_bert_word_embedder()

        model_function = self._find_model_function_pure()
        model = model_function(
            word_embedder,
            position_embedder,
            self.vocab,
            self.configuration,
            bert_word_embedder=bert_word_embedder
        )

        self._print_args(model)
        model = model.to(self.configuration['device'])
        return model

    def _get_optimizer(self, model):
        _params = filter(lambda p: p.requires_grad, model.parameters())
        if self.configuration['fixed_bert']:
            return optim.Adam(_params, lr=0.001, weight_decay=0.00001)
        else:
            return optim.Adam(_params, lr=self.configuration['learning_rate_in_bert'],
                              weight_decay=self.configuration['l2_in_bert'])

    def _get_position_embeddings_dim(self):
        return self.configuration['position_embeddings_dim']


class DatasetReaderForNerBert(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 position_indexers: Dict[str, TokenIndexer] = None,
                 core_nlp: my_corenlp.StanfordCoreNLP=None,
                 configuration=None, sentence_segmenter: BaseSentenceSegmenter=NltkSentenceSegmenter(),
                 bert_tokenizer=None,
                 bert_token_indexers=None
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        self.bert_tokenizer = bert_tokenizer
        self.bert_token_indexers = bert_token_indexers or {"bert": SingleIdTokenIndexer(namespace="bert")}
        self.position_indexers = position_indexers or {"position": SingleIdTokenIndexer(namespace='position')}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.core_nlp = core_nlp
        self.configuration = configuration
        self.sentence_segmenter = sentence_segmenter

    @overrides
    def text_to_instance(self, samples: List) -> Instance:
        sample = samples[0]
        fields = {}

        words: List = sample['words']
        sample['words'] = words

        sample['length'] = len(words)

        tokens = [Token(word.lower()) for word in words]
        fields['tokens'] = TextField(tokens, self.token_indexers)

        bert_words = ['[CLS]']
        word_index_and_bert_indices = {}
        for i, word in enumerate(words):
            bert_ws = self.bert_tokenizer.tokenize(word.lower())
            word_index_and_bert_indices[i] = []
            for j in range(len(bert_ws)):
                word_index_and_bert_indices[i].append(len(bert_words) + j)
            bert_words.extend(bert_ws)
        bert_words.append('[SEP]')
        bert_tokens = [Token(word) for word in bert_words]
        bert_text_field = TextField(bert_tokens, self.bert_token_indexers)
        fields['bert'] = bert_text_field
        sample['bert_words'] = bert_words
        sample['word_index_and_bert_indices'] = word_index_and_bert_indices

        position = []
        for i in range(len(words)):
            position.append(Token(str(i)))
        position_field = TextField(position, self.position_indexers)
        fields['position'] = position_field

        if 'target_tags' in sample:
            # target_part: str = sample['metadata']['original_line'].split('####')[1]
            # tags = [e.split('=')[1] for e in target_part.split(' ')]
            # for i, tag in enumerate(tags):
            #     if not tag.startswith('T'):
            #         tags[i] = 'O'
            #     else:
            #         if i == 0 or tags[i - 1] == 'O':
            #             tags[i] = 'B'
            #         else:
            #             tags[i] = 'I'

            tags = copy.deepcopy(sample['target_tags'])
            for sample_temp in samples[1:]:
                tags_temp = sample_temp['target_tags']
                for i, tag_temp in enumerate(tags_temp):
                    if tag_temp != 'O':
                        tags[i] = tag_temp

            sample['all_target_tags'] = tags
            # for evaluation using the estimator of TOWE
            sample['opinion_words_tags'] = tags
            fields["labels"] = SequenceLabelField(
                labels=tags, sequence_field=fields['tokens'], label_namespace='target_tags'
            )

        sample_field = MetadataField(sample)
        fields["sample"] = sample_field
        return Instance(fields)

    @overrides
    def _read(self, samples: list) -> Iterator[Instance]:
        # unique_original_lines = set()
        # for sample in samples:
        #     if sample['metadata']['original_line'] in unique_original_lines:
        #         continue
        #     else:
        #         unique_original_lines.add(sample['metadata']['original_line'])
        #     yield self.text_to_instance(sample)
        sentence_and_samples = OrderedDict()
        for sample in samples:
            words = sample['words']
            sentence = ' '.join(words)
            if sentence not in sentence_and_samples:
                sentence_and_samples[sentence] = []
            sentence_and_samples[sentence].append(sample)
        for samples in sentence_and_samples.values():
            yield self.text_to_instance(samples)


def add_bert_words_and_word_index_bert_indices(self, words, fields, sample):
    bert_words = ['[CLS]']
    word_index_and_bert_indices = {}
    for i, word in enumerate(words):
        bert_ws = self.bert_tokenizer.tokenize(word.lower())
        word_index_and_bert_indices[i] = []
        for j in range(len(bert_ws)):
            word_index_and_bert_indices[i].append(len(bert_words) + j)
        bert_words.extend(bert_ws)
    bert_words.append('[SEP]')
    bert_tokens = [Token(word) for word in bert_words]
    bert_text_field = TextField(bert_tokens, self.token_indexers)
    fields['bert'] = bert_text_field
    sample['bert_words'] = bert_words
    sample['word_index_and_bert_indices'] = word_index_and_bert_indices


class NerBert(SequenceLabelingModel):
    def __init__(self, word_embedder: TextFieldEmbedder, position_embedder: TextFieldEmbedder,
                 vocab: Vocabulary, configuration: dict, bert_word_embedder: TextFieldEmbedder=None):
        super().__init__(vocab)
        self.configuration = configuration

        self.word_embedder = word_embedder
        self.bert_word_embedder = bert_word_embedder
        self.position_embedder = position_embedder

        self.embedding_dim = word_embedder.get_output_dim()
        self.bert_embedding_dim = self.bert_word_embedder.get_output_dim()
        self.position_dim = self.position_embedder.get_output_dim()

        if self.configuration['position']:
            self.lstm_input_size = self.bert_embedding_dim + self.position_dim
        else:
            self.lstm_input_size = self.bert_embedding_dim
        self.hidden_size = self.bert_embedding_dim // 2
        if self.configuration['lstm_layer_num_in_bert'] != 0:
            self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size,
                                num_layers=self.configuration['lstm_layer_num_in_bert'],
                                bidirectional=True, batch_first=True)

        self.feedforward = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

        if self.configuration['crf']:
            # BIO、BIOSE、IOB、BILOU、BMEWO、BMEWO+ https://zhuanlan.zhihu.com/p/147537898
            tagger_ner: CrfTagger = CrfTagger(vocab=vocab,
                                              output_dim=self.hidden_size * 2,
                                              label_namespace='target_tags',
                                              # label_encoding='BIO',
                                              # constrain_crf_decoding=True,
                                              dropout=None,
                                              regularizer=None
                                              )
        else:
            tagger_ner: SimpleTagger = SimpleTagger(vocab=vocab,
                                                    output_dim=self.hidden_size * 2,
                                                    label_namespace='target_tags',
                                                    regularizer=None
                                                    )
        self._tagger_ner = tagger_ner

        self.dropout = nn.Dropout(0.5)

    def forward(self, tokens: Dict[str, torch.Tensor], position: torch.Tensor, sample: list,
                labels: torch.Tensor=None, bert: torch.Tensor=None) -> torch.Tensor:
        embedded_text_input = self.word_embedder(tokens)
        word_embeddings_size = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)

        bert_mask = bert['mask']
        # bert_word_embeddings = self.bert_word_embedder(bert)
        token_type_ids = bert['bert-type-ids']
        # token_type_ids_size = token_type_ids.size()
        # for i in range(token_type_ids_size[1]):
        #     print(token_type_ids[0][i])
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)

        if self.configuration['position']:
            position_input = self.position_embedder(position)
            lstm_input = torch.cat([aspect_word_embeddings_from_bert_cat, position_input], dim=-1)
        else:
            lstm_input = aspect_word_embeddings_from_bert_cat

        lstm_input = self.dropout(lstm_input)

        if self.configuration['lstm_layer_num_in_bert'] != 0:
            lstm_result, _ = self.lstm(lstm_input)
            lstm_result = self.dropout(lstm_result)
        else:
            lstm_result = lstm_input

        lstm_result = self.dropout(lstm_result)

        encoded_text = self.feedforward(lstm_result)
        encoded_text = self.dropout(encoded_text)

        input_for_crf_tagger = {
            'encoded_text': encoded_text,
            'mask': mask,
            'tags': labels,
            'metadata': sample
        }
        result = self._tagger_ner.forward(**input_for_crf_tagger)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._tagger_ner.get_metrics(reset=reset)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        result = self._tagger_ner.decode(output_dict)
        return result


    def get_bert_embedding(self, bert, sample, word_embeddings_size):
        bert_mask = bert['mask']
        token_type_ids = bert['bert-type-ids']
        offsets = bert['bert-offsets']
        bert_word_embeddings = self.bert_word_embedder(bert, token_type_ids=token_type_ids, offsets=offsets)

        aspect_word_embeddings_from_bert = []
        for j in range(len(sample)):
            aspect_word_embeddings_from_bert_of_one_sample = []
            all_word_indices_in_bert = sample[j]['word_index_and_bert_indices']
            for k in range(word_embeddings_size[1]):
                is_index_greater_than_max_len = False
                if k in all_word_indices_in_bert:
                    for index in all_word_indices_in_bert[k]:
                        if index >= self.configuration['max_len']:
                            is_index_greater_than_max_len = True
                            break
                if not is_index_greater_than_max_len and k in all_word_indices_in_bert:
                    word_indices_in_bert = all_word_indices_in_bert[k]
                    word_bert_embeddings = []
                    for word_index_in_bert in word_indices_in_bert:
                        word_bert_embedding = bert_word_embeddings[j][word_index_in_bert]
                        word_bert_embeddings.append(word_bert_embedding)
                    if len(word_bert_embeddings) == 0:
                        print()
                    if len(word_bert_embeddings) > 1:
                        word_bert_embeddings_unsqueeze = [torch.unsqueeze(e, dim=0) for e in word_bert_embeddings]
                        word_bert_embeddings_cat = torch.cat(word_bert_embeddings_unsqueeze, dim=0)
                        word_bert_embeddings_sum = torch.sum(word_bert_embeddings_cat, dim=0)
                        word_bert_embeddings_ave = word_bert_embeddings_sum / len(word_bert_embeddings)
                    else:
                        word_bert_embeddings_ave = word_bert_embeddings[0]
                    aspect_word_embeddings_from_bert_of_one_sample.append(
                        torch.unsqueeze(word_bert_embeddings_ave, 0))
                else:
                    zero = torch.zeros_like(torch.unsqueeze(bert_word_embeddings[0][0], 0))
                    aspect_word_embeddings_from_bert_of_one_sample.append(zero)
            aspect_word_embeddings_from_bert_of_one_sample_cat = torch.cat(
                aspect_word_embeddings_from_bert_of_one_sample, dim=0)
            aspect_word_embeddings_from_bert.append(
                torch.unsqueeze(aspect_word_embeddings_from_bert_of_one_sample_cat, dim=0))
        aspect_word_embeddings_from_bert_cat = torch.cat(aspect_word_embeddings_from_bert, dim=0)
        return aspect_word_embeddings_from_bert_cat