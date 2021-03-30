# -*- coding: utf-8 -*-


from allennlp.models import Model

from nlp_tasks.absa.mining_opinions.sequence_labeling import pytorch_models


class Callback(object):
    """Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def on_epoch_end(self, epoch: int):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_batch_end(self, batch: int):
        pass

    def on_train_begin(self):
        pass


class EstimateCallback(Callback):

    def __init__(self, data_type_and_data: dict, estimator: pytorch_models.Estimator, logger):
        self.data_type_and_data = data_type_and_data
        self.estimator = estimator
        self.logger = logger

    def on_epoch_end(self, epoch):
        for data_type, data in self.data_type_and_data.items():
            result = self.estimator.estimate(data)
            self.logger.info('epoch: %d data_type: %s result: %s' % (epoch, data_type, str(result)))

    def on_batch_end(self, batch: int):
        for data_type, data in self.data_type_and_data.items():
            result = self.estimator.estimate(data)
            self.logger.info('batch: %d data_type: %s result: %s' % (batch, data_type, str(result)))


class SetLossWeightCallback(Callback):

    def __init__(self, model, logger, acd_warmup_epoch_num=0):
        self.model = model
        self.acd_warmup_epoch_num = acd_warmup_epoch_num
        self.logger = logger

    def on_epoch_begin(self, epoch: int):
        if epoch < self.acd_warmup_epoch_num:
            self.model.sentiment_loss_weight = 0
        else:
            self.model.category_loss_weight = 1
            self.model.sentiment_loss_weight = 1


class FixedLossWeightCallback(Callback):

    def __init__(self, model, logger, category_loss_weight=1,
                 sentiment_loss_weight=1):
        self.model = model
        self.category_loss_weight = category_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight

    def on_train_begin(self):
        self.model.category_loss_weight = self.category_loss_weight
        self.model.sentiment_loss_weight = self.sentiment_loss_weight


class LossWeightCallback(Callback):

    def __init__(self, model, logger, loss_weights:dict):
        self.model = model
        self.loss_weights = loss_weights

    def on_train_begin(self):
        self.model.loss_weights.update(self.loss_weights)
