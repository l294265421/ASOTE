# -*- coding: utf-8 -*-

import copy
import logging
import sys

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from nlp_tasks.absa.conf import model_path, data_path, thresholds, datasets
from nlp_tasks.absa.models import keras_layers
from nlp_tasks.absa.preprocess import label_mapping
from nlp_tasks.absa.utils import file_utils
from nlp_tasks.absa.utils import model_utils, evaluate_utils
from nlp_tasks.absa.utils import common_utils


class F1(Callback):
    def __init__(self, validation_data=(), interval=1, num_class=3):
        super(F1, self).__init__()

        self.interval = interval
        self.num_class = num_class
        self.X_val, self.y_val, self.X_tra, self.y_tra = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred_tra = self.model.predict(self.X_tra, verbose=0)
            threshold = [1.0] * datasets.sentiment_class_num
            if self.num_class != 3:
                threshold = thresholds.topic_positive_threshold
            precision_tra = evaluate_utils.precision(evaluate_utils.to_list(self.y_tra),
                                                     evaluate_utils.to_list(y_pred_tra), threshold)
            recall_tra = evaluate_utils.recall(evaluate_utils.to_list(self.y_tra),
                                               evaluate_utils.to_list(y_pred_tra), threshold)
            f1_tra = evaluate_utils.f1(evaluate_utils.to_list(self.y_tra),
                                       evaluate_utils.to_list(y_pred_tra), threshold)
            print("\n train - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                  % (epoch + 1, precision_tra, recall_tra, f1_tra))

            y_pred = self.model.predict(self.X_val, verbose=0)
            precision = evaluate_utils.precision(evaluate_utils.to_list(self.y_val),
                                                 evaluate_utils.to_list(y_pred), threshold)
            recall = evaluate_utils.recall(evaluate_utils.to_list(self.y_val),
                                           evaluate_utils.to_list(y_pred), threshold)
            f1 = evaluate_utils.f1(evaluate_utils.to_list(self.y_val),
                                   evaluate_utils.to_list(y_pred), threshold)
            print("\n val - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                  % (epoch + 1, precision, recall, f1))
            logs['f1'] = f1


class joint_model_F1(Callback):
    def __init__(self, validation_data=(), interval=1, num_class=3):
        super(joint_model_F1, self).__init__()

        self.interval = interval
        self.num_class = num_class
        self.X_test, self.y_test, self.X_val, self.y_val, self.X_tra, self.y_tra = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            predict_datas = {
                'train': (self.X_tra, self.y_tra),
                'val': (self.X_val, self.y_val),
                'test': (self.X_test, self.y_test)
            }
            for data_category, (X, y) in predict_datas.items():
                y_pred = self.model.predict(X, verbose=0)
                f1_aspect = evaluate_utils.evaluate_aspect_of_joint_model(y_pred, y,
                                                              '%s aspect' % data_category, epoch)
                acc_sentiment, f1_sentiment = evaluate_utils.evaluate_sentiment_of_joint_model(y_pred, y,
                                                                 '%s sentiment' % data_category, epoch)
                prefix = '' if data_category == 'val' else data_category + '_'
                logs[prefix + 'f1_aspect'] = f1_aspect
                logs[prefix + 'f1_sentiment'] = f1_sentiment
                logs[prefix + 'acc_sentiment'] = acc_sentiment
                float('%.4f' % f1_sentiment)
                logs[prefix + 'total'] = float('%.4f' % f1_sentiment) * 100000 * 100000 + float(
                    '%.4f' % acc_sentiment) * 100000 + float('%.4f' % f1_aspect)

    def on_batch_end(self, batch, logs=None):
        l = logs['loss']


class MultiLabelF1(Callback):
    def __init__(self, data, interval=1, num_class=3, logger=None):
        super().__init__()

        self.interval = interval
        self.num_class = num_class
        self.data = data
        self.logger = logger

    @staticmethod
    def evaluate(model, X, y, threshold=0.5):
        """

        :param model:
        :param X:
        :param y:
        :return:
        """
        y_pred_prob = model.predict(X, verbose=0)
        y_pred = np.concatenate(y_pred_prob, axis=1)
        y_pred = y_pred >= threshold
        y_pred = y_pred.astype(np.int)
        precision = precision_score(y, y_pred, average='micro')
        recall = recall_score(y, y_pred, average='micro')
        f1 = f1_score(y, y_pred, average='micro')
        return precision, recall, f1

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            predict_datas = self.data
            performances = {}
            for data_category, (X, y) in predict_datas.items():
                precision, recall, f1 = MultiLabelF1.evaluate(self.model, X, y)
                prefix = '' if data_category == 'val' else data_category + '_'
                performances[prefix + 'precision_visual'] = precision
                performances[prefix + 'recall_visual'] = recall
                performances[prefix + 'f1_visual'] = f1
            logs.update(performances)
            self.logger.info('%d-%s' % (epoch + 1, str(performances)))


class AcdTwoStagesMultiLabelF1(Callback):
    def __init__(self, data, entity_aspect_pair_num, entity_num, interval=1, logger=None):
        super().__init__()

        self.interval = interval
        self.entity_aspect_pair_num = entity_aspect_pair_num
        self.entity_num = entity_num
        self.data = data
        self.logger = logger

    @staticmethod
    def evaluate_multi_label(y_true, y_pred_prob, threshold=0.5):
        y_true = [np.expand_dims(e, axis=1) for e in y_true]
        y_true = np.concatenate(y_true, axis=1)

        y_pred = np.concatenate(y_pred_prob, axis=1)
        y_pred = y_pred >= threshold
        y_pred = y_pred.astype(np.int)

        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        return precision, recall, f1

    @staticmethod
    def evaluate(model, X, y, entity_aspect_pair_num, entity_num, threshold=0.5):
        """

        :param model:
        :param X:
        :param y:
        :return:
        """
        y_pred_prob = model.predict(X, verbose=0)

        y_true_aspect = y[: entity_aspect_pair_num]
        y_pred_prob_aspect = y_pred_prob[: entity_aspect_pair_num]
        aspect_precision, aspect_recall, aspect_f1 = AcdTwoStagesMultiLabelF1.evaluate_multi_label(y_true_aspect,
                                                                                                   y_pred_prob_aspect)

        y_true_entity = y[entity_aspect_pair_num:]
        y_pred_prob_entity = y_pred_prob[entity_aspect_pair_num:]
        entity_precision, entity_recall, entity_f1 = AcdTwoStagesMultiLabelF1.evaluate_multi_label(y_true_entity,
                                                                                                   y_pred_prob_entity)

        return aspect_precision, aspect_recall, aspect_f1, entity_precision, entity_recall, entity_f1

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            predict_datas = self.data
            performances = {}
            for data_category, (X, y) in predict_datas.items():
                aspect_precision, aspect_recall, aspect_f1, entity_precision, entity_recall, entity_f1 = \
                    AcdTwoStagesMultiLabelF1.evaluate(self.model, X, y, self.entity_aspect_pair_num, self.entity_num)
                prefix = '' if data_category == 'val' else data_category + '_'
                performances[prefix + 'aspect_precision_visual'] = aspect_precision
                performances[prefix + 'aspect_recall_visual'] = aspect_recall
                performances[prefix + 'aspect_f1_visual'] = aspect_f1
                performances[prefix + 'entity_precision_visual'] = entity_precision
                performances[prefix + 'entity_recall_visual'] = entity_recall
                performances[prefix + 'entity_f1_visual'] = entity_f1
            logs.update(performances)
            self.logger.info('%d-%s' % (epoch + 1, str(performances)))


class MultiClassF1(Callback):
    def __init__(self, data, interval=1, num_class=3):
        super().__init__()

        self.interval = interval
        self.num_class = num_class
        self.data = data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            predict_datas = self.data
            performances = {}
            for data_category, (X, y) in predict_datas.items():
                y_pred_prob = self.model.predict(X, verbose=0)
                y_pred = []
                for i in range(y_pred_prob.shape[0]):
                    one_pred_prob = y_pred_prob[i]
                    max_index = np.argmax(one_pred_prob, axis=0)
                    one_pred = [0] * self.num_class
                    one_pred[max_index] = 1
                    y_pred.append(one_pred)

                y_pred = np.array(y_pred)
                precision = precision_score(y, y_pred, average='macro')
                recall = recall_score(y, y_pred, average='macro')
                f1 = f1_score(y, y_pred, average='macro')
                prefix = '' if data_category == 'val' else data_category + '_'
                performances[prefix + 'precision_visual'] = precision
                performances[prefix + 'recall_visual'] = recall
                performances[prefix + 'f1_visual'] = f1
            logs.update(performances)
            logging.info('%d-%s' % (epoch, str(performances)))


class AcdAndScMetrics(Callback):
    def __init__(self, data, logger, interval=1, category_num=10, polarity_num=3, threshold=0.5):
        super().__init__()
        self.logger = logger
        self.interval = interval
        self.category_num = category_num
        self.polarity_num = polarity_num
        self.data = data
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            predict_datas = self.data
            performances = {}
            for data_category, (X, y) in predict_datas.items():
                y_pred_prob = self.model.predict(X, verbose=0)

                # category
                y_category = y[: self.category_num]
                y_category = [e[:, np.newaxis] for e in y_category]
                y_category = np.concatenate(y_category, axis=1)

                y_pred_prob_category = y_pred_prob[: self.category_num]
                y_pred_category = np.concatenate(y_pred_prob_category, axis=1)
                y_pred_category = y_pred_category >= self.threshold
                y_pred_category = y_pred_category.astype(np.int)
                precision_category = precision_score(y_category, y_pred_category, average='micro')
                recall_category = recall_score(y_category, y_pred_category, average='micro')
                f1_category = f1_score(y_category, y_pred_category, average='micro')
                prefix = '' if data_category == 'val' else data_category + '_'
                performances[prefix + 'precision_category_visual'] = precision_category
                performances[prefix + 'recall_category_visual'] = recall_category
                performances[prefix + 'f1_category_visual'] = f1_category

                # sentiment
                y_sentiment = y[self.category_num:]
                y_sentiment = [np.expand_dims(e, axis=1)for e in y_sentiment]
                y_sentiment = np.concatenate(y_sentiment, axis=1)
                y_pred_prob_sentiment = y_pred_prob[self.category_num:]
                y_pred_prob_sentiment = [np.expand_dims(e, axis=1) for e in y_pred_prob_sentiment]
                y_pred_prob_sentiment = np.concatenate(y_pred_prob_sentiment, axis=1)

                y_sc_true = []
                y_sc_pred = []
                y_acd_sc_true = []
                y_acd_sc_pred = []
                shape = y_sentiment.shape
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        category_indicator_true = y_category[i][j]
                        category_indicator_pred = y_pred_category[i][j]

                        one_y_sc_true = y_sentiment[i][j]
                        one_y_acd_sc_true_list = copy.deepcopy(one_y_sc_true.tolist())
                        if category_indicator_true == 1:
                            one_y_acd_sc_true_list.append(0)
                        else:
                            one_y_acd_sc_true_list = [0] * (self.polarity_num + 1)
                            # one_y_acd_sc_true_list[-1] = 1
                        y_acd_sc_true.append(one_y_acd_sc_true_list)

                        one_y_sc_pred = y_pred_prob_sentiment[i][j]
                        max_index = np.argmax(one_y_sc_pred, axis=0)
                        one_y_sc_pred_list = [0] * self.polarity_num
                        one_y_sc_pred_list[max_index] = 1
                        one_y_acd_sc_pred_list = copy.deepcopy(one_y_sc_pred_list)
                        if category_indicator_pred == 1:
                            one_y_acd_sc_pred_list.append(0)
                        else:
                            one_y_acd_sc_pred_list = [0] * (self.polarity_num + 1)
                            # one_y_acd_sc_pred_list[-1] = 1
                        y_acd_sc_pred.append(one_y_acd_sc_pred_list)

                        if category_indicator_true == 1:
                            y_sc_true.append(one_y_sc_true)
                            y_sc_pred.append(one_y_sc_pred_list)

                y_sc_true = np.array(y_sc_true)
                y_sc_pred = np.array(y_sc_pred)
                precision_category = accuracy_score(y_sc_true, y_sc_pred)
                performances[prefix + 'acc_sc_visual'] = precision_category

                y_acd_sc_true = np.array(y_acd_sc_true)
                y_acd_sc_pred = np.array(y_acd_sc_pred)
                precision_acd_sc = precision_score(y_acd_sc_true, y_acd_sc_pred, average='micro')
                recall_acd_sc = recall_score(y_acd_sc_true, y_acd_sc_pred, average='micro')
                f1_acd_sc = f1_score(y_acd_sc_true, y_acd_sc_pred, average='micro')
                performances[prefix + 'precision_acd_sc_visual'] = precision_acd_sc
                performances[prefix + 'recall_acd_sc_visual'] = recall_acd_sc
                performances[prefix + 'f1_acd_sc_visual'] = f1_acd_sc
            logs.update(performances)
            self.logger.info('%d-%s' % (epoch, str(performances)))


class AcdAndDomainMetrics(Callback):
    def __init__(self, data, interval=1, category_num=10, threshold=0.5):
        super().__init__()

        self.interval = interval
        self.category_num = category_num
        self.data = data
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            predict_datas = self.data
            performances = {}
            for data_category, (X, y) in predict_datas.items():
                y_pred_prob = self.model.predict(X, verbose=0)

                # category
                y_category = y[: self.category_num]
                y_category = [e[:, np.newaxis] for e in y_category]
                y_category = np.concatenate(y_category, axis=1)

                y_pred_prob_category = y_pred_prob[: self.category_num]
                y_pred_category = np.concatenate(y_pred_prob_category, axis=1)
                y_pred_category = y_pred_category >= self.threshold
                y_pred_category = y_pred_category.astype(np.int)
                precision_category = precision_score(y_category, y_pred_category, average='micro')
                recall_category = recall_score(y_category, y_pred_category, average='micro')
                f1_category = f1_score(y_category, y_pred_category, average='micro')
                prefix = '' if data_category == 'val' else data_category + '_'
                performances[prefix + 'precision_category_visual'] = precision_category
                performances[prefix + 'recall_category_visual'] = recall_category
                performances[prefix + 'f1_category_visual'] = f1_category

                # domain
                y_domain = y[self.category_num]
                y_pred_prob_domain= y_pred_prob[self.category_num]
                y_pred_domain = []
                for i in range(y_pred_prob_domain.shape[0]):
                    one_pred_prob = y_pred_prob_domain[i]
                    max_index = np.argmax(one_pred_prob, axis=0)
                    one_pred = [0] * len(one_pred_prob)
                    one_pred[max_index] = 1
                    y_pred_domain.append(one_pred)

                y_pred_domain = np.array(y_pred_domain)
                precision_domain = precision_score(y_domain, y_pred_domain, average='macro')
                recall_domain = recall_score(y_domain, y_pred_domain, average='macro')
                f1_domain = f1_score(y_domain, y_pred_domain, average='macro')
                performances[prefix + 'precision_domain_visual'] = precision_domain
                performances[prefix + 'recall_domain_visual'] = recall_domain
                performances[prefix + 'f1_domain_visual'] = f1_domain

            logs.update(performances)
            logging.info('%d-%s' % (epoch, str(performances)))



def cv_multi_input(x_train, y_train, k, x_test, x_val, epochs, monitor_threshold, model_name,
                   pretrain_model_path, batch_size, is_one_fold, sample_weight, num_class,
                   get_model_fun, *model_fun_arg):
    """cv_multi_input"""
    result = []
    result_val = []
    ntrain = x_train[0].shape[0]
    oof_train = np.zeros((ntrain, num_class))
    kf = KFold(n_splits=k, shuffle=True)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train[0], y_train):
        X_tra = []
        X_val = []
        for i in range(len(x_train)):
            X_tra.append(x_train[i][train_index])
            X_val.append(x_train[i][test_index])
        y_tra = y_train[train_index]
        y_val = y_train[test_index]

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = F1(validation_data=(X_val, y_val, X_tra, y_tra), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(
            serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        if sample_weight is not None:
            sample_weight_temp = sample_weight[train_index]
        else:
            sample_weight_temp = None
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks_list, sample_weight=sample_weight_temp)
        # oof_train[test_index] = model.predict(X_val)
        if checkpoint.best < monitor_threshold:
            continue
        print('best f1: %s' % str(checkpoint.best))
        bests.append(checkpoint.best)
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_val is not None:
            y_val_pred = model.predict(x_val, batch_size=1024)
            result_val.append(y_val_pred)

        oof_train[test_index] = model.predict(X_val)
        serial_num += 1
        if is_one_fold:
            break

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)
    y_test = average_of_pred(result)
    y_val = average_of_pred(result_val)

    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, y_val


def subset(ndarray_list, index):
    result = []
    for i in range(len(ndarray_list)):
        result.append(ndarray_list[i][index])
    return result


def cv_multi_input_multi_output_joint_model(x_train, y_train, k, x_test, x_val, epochs, monitor_threshold, model_name,
                   pretrain_model_path, batch_size, is_one_fold, sample_weight, num_class,
                   get_model_fun, *model_fun_arg):
    """cv_multi_input"""
    result = []
    result_val = []
    kf = KFold(n_splits=k, shuffle=True)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train[0]):
        X_tra = subset(x_train, train_index)
        X_val = subset(x_train, test_index)
        y_tra = subset(y_train, train_index)
        y_val = subset(y_train, test_index)

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = joint_model_F1(validation_data=(X_val, y_val, X_tra, y_tra), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=10, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(
            serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        if sample_weight is not None:
            sample_weight_temp = sample_weight[train_index]
        else:
            sample_weight_temp = None
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks_list, sample_weight=sample_weight_temp)
        # oof_train[test_index] = model.predict(X_val)
        if checkpoint.best < monitor_threshold:
            continue
        print('best f1: %s' % str(checkpoint.best))
        bests.append(checkpoint.best)
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        result.append(y_pred)

        if x_val is not None:
            y_val_pred = model.predict(x_val, batch_size=1024)
            result_val.append(y_val_pred)

        serial_num += 1
        if is_one_fold:
            break

    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    return average_of_prediction_of_muliti_out(result), average_of_prediction_of_muliti_out(result_val)


def cv_provide_train_val(x_train, y_train, x_val, y_val, epochs, monitor, pretrain_model_path,
                         batch_size, best_model_filepath, sample_weight, model, model_log_dir,
                         customized_callbacks=None, patience=10):
    """cv_multi_input"""
    if sys.platform.startswith('win'):
        plot_model(model, to_file=best_model_filepath + '.png')
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)
    mode = 'max'
    early_stopping = EarlyStopping(monitor=monitor, min_delta=0.00001, patience=patience, verbose=0,
                                   mode=mode)
    checkpoint = ModelCheckpoint(best_model_filepath, monitor=monitor, verbose=0,
                                 save_best_only=False, save_weights_only=True,
                                 mode=mode, period=1)
    callbacks_list = []
    if customized_callbacks:
        callbacks_list.extend(customized_callbacks)
    callbacks_list.extend([early_stopping, checkpoint])
    if sys.platform.startswith('win'):
        tb = TensorBoard(log_dir=model_log_dir, write_graph=True)
        callbacks_list.append(tb)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(x_val, y_val),
              callbacks=callbacks_list, sample_weight=sample_weight, shuffle=True)
    logging.info('%s %.4f' % ('best', checkpoint.best))


def cv_multi_input_val_loss(x_train, y_train, k, x_test, x_val, epochs, monitor_threshold, model_name,
                   pretrain_model_path, batch_size, is_one_fold, sample_weight,
                   get_model_fun, *model_fun_arg):
    """cv_multi_input"""
    result = []
    result_val = []
    ntrain = x_train[0].shape[0]
    oof_train = np.zeros((ntrain, 3))
    kf = KFold(n_splits=k, shuffle=True)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train[0], y_train):
        X_tra = []
        X_val = []
        for i in range(len(x_train)):
            X_tra.append(x_train[i][train_index])
            X_val.append(x_train[i][test_index])
        y_tra = y_train[train_index]
        y_val = y_train[test_index]

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = F1(validation_data=(X_val, y_val, X_tra, y_tra), interval=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=4, verbose=0,
                                       mode='min')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(
            serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='val_loss', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='min', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        if sample_weight is not None:
            sample_weight_temp = sample_weight[train_index]
        else:
            sample_weight_temp = None
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks_list, sample_weight=sample_weight_temp)
        # oof_train[test_index] = model.predict(X_val)
        # if checkpoint.best < monitor_threshold:
        #     continue
        print('best f1: %s' % str(checkpoint.best))
        bests.append(checkpoint.best)
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_val is not None:
            y_val_pred = model.predict(x_val, batch_size=1024)
            result_val.append(y_val_pred)

        oof_train[test_index] = model.predict(X_val)
        serial_num += 1
        if is_one_fold:
            break

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)
    y_test = average_of_pred(result)
    y_val = average_of_pred(result_val)

    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, y_val


def cv_multi_input_external(x_train, y_train, k, x_test, x_val,
                            x_external, y_external,
                            epochs, monitor_threshold, model_name,
                            pretrain_model_path, batch_size,
                            get_model_fun, *model_fun_arg):
    """cv_multi_input"""
    result = []
    result_val = []
    ntrain = x_train[0].shape[0]
    oof_train = np.zeros((ntrain, 3))
    kf = KFold(n_splits=k, shuffle=True, random_state=223)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train[0], y_train):
        X_tras = []
        X_val = []
        for i in range(len(x_train)):
            X_tra = x_train[i][train_index]
            if x_external is not None:
                X_tra = np.vstack([x_external[i], X_tra])
            X_tras.append(X_tra)
            X_val.append(x_train[i][test_index])
        y_tra = y_train[train_index]
        if y_external is not None:
            y_tra = np.vstack((y_external, y_tra))
        y_val = y_train[test_index]

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = F1(validation_data=(X_val, y_val, X_tras, y_tra), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(
            serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        model.fit(X_tras, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks_list)
        # oof_train[test_index] = model.predict(X_val)
        # if checkpoint.best < monitor_threshold:
        #     continue
        print('best f1: %s' % str(checkpoint.best))
        bests.append(checkpoint.best)
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_val is not None:
            y_val_pred = model.predict(x_val, batch_size=1024)
            result_val.append(y_val_pred)

        oof_train[test_index] = model.predict(X_val)
        serial_num += 1

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)
    y_test = average_of_pred(result)
    y_val = average_of_pred(result_val)

    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, y_val


def bagging_multi_input(x_train, y_train, k, x_test, epochs, monitor_threshold, model_name,
                   pretrain_model_path, batch_size,
                   get_model_fun, *model_fun_arg):
    """cv_multi_input"""
    result = []
    # ntrain = x_train.shape[0]
    # oof_train = np.zeros((ntrain, class_num))
    kf = KFold(n_splits=k, shuffle=True, random_state=223)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train[0], y_train):
        X_tra = []
        X_val = []
        for i in range(len(x_train)):
            X_tra.append(x_train[i][train_index])
            X_val.append(x_train[i][test_index])
        y_tra = y_train[train_index]
        y_val = y_train[test_index]

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = F1(validation_data=(X_val, y_val, X_tra, y_tra), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + str(serial_num) + '.hdf5'
        serial_num += 1
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks_list)
        # oof_train[test_index] = model.predict(X_val)
        if checkpoint.best < monitor_threshold:
            continue
        print('best f1: %s' % str(checkpoint.best))
        bests.append(checkpoint.best)
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        result.append(y_pred)

    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)
    y_test = result[0]
    for i in range(1, len(result)):
        y_test += result[i]
    y_test /= len(result)
    return y_test


def cv_one_fold(x_train, y_train, train_ratio, x_test, x_val, epochs, model_name, pretrain_model_path,
                batch_size, get_model_fun, *model_fun_arg):
    """cv_one_fold"""
    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=train_ratio,
                                                  random_state=233,
                                                  stratify=evaluate_utils.one_hot_to_label(y_train))

    model = get_model_fun(*model_fun_arg)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)
    f1 = F1(validation_data=(X_val, y_val, X_tra, y_tra), interval=1)
    early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                   mode='max')
    best_model_filepath = model_path.model_file_dir + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                 save_best_only=True, save_weights_only=True,
                                 mode='max', period=1)
    # log_dir = model_path.model_log_dir + 'cv_multi_output_one_fold'
    # tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
    # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [f1, early_stopping, checkpoint]
    model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
              callbacks=callbacks_list)
    print('best f1: %s' % str(checkpoint.best))
    model.load_weights(best_model_filepath)
    y_pred = model.predict(x_test, batch_size=1024)
    y_val_pred = model.predict(x_val, batch_size=1024)
    if x_val is not None:
        y_val_pred = model.predict(x_val, batch_size=1024)
    return y_pred, y_val_pred


def cv_multi_input_one_fold(x_train, y_train, train_ratio, x_test, x_val, epochs, model_name,
                            pretrain_model_path, batch_size, get_model_fun, *model_fun_arg):
    """cv_multi_input_one_fold"""
    X_tras = []
    X_vals = []
    for i in range(len(x_train)):
        X_tra, X_val, y_tra, y_val = train_test_split(x_train[i], y_train, train_size=train_ratio,
                                                      random_state=233,
                                                      stratify=evaluate_utils.one_hot_to_label(y_train))
        X_tras.append(X_tra)
        X_vals.append(X_val)

    model = get_model_fun(*model_fun_arg)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)

    y_pred = model.predict(X_vals, verbose=0)
    precision = evaluate_utils.precision(evaluate_utils.to_list(y_val),
                                         evaluate_utils.to_list(y_pred), 0.6)
    recall = evaluate_utils.recall(evaluate_utils.to_list(y_val),
                                   evaluate_utils.to_list(y_pred), 0.6)
    f1 = evaluate_utils.f1(evaluate_utils.to_list(y_val),
                           evaluate_utils.to_list(y_pred), 0.6)
    print("\n val - precision: %.6f - recall: %.6f - f1: %.6f \n"
          % (precision, recall, f1))

    f1 = F1(validation_data=(X_vals, y_val, X_tras, y_tra), interval=1)
    early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                   mode='max')
    best_model_filepath = model_path.model_file_dir + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                 save_best_only=True, save_weights_only=True,
                                 mode='max', period=1)
    # log_dir = model_path.model_log_dir + 'cv_multi_output_one_fold'
    # tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
    # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [f1, early_stopping, checkpoint]
    model.fit(X_tras, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
              callbacks=callbacks_list, class_weight='auto')
    print('best f1: %s' % str(checkpoint.best))
    model.load_weights(best_model_filepath)
    y_pred = model.predict(x_test, batch_size=1024)
    y_val_pred = None
    if x_val is not None:
        y_val_pred = model.predict(x_val, batch_size=1024)
    return y_pred, y_val_pred


def cv_multi_input_one_fold_external(x_train, y_train, train_ratio, x_test, x_val,
                                     x_external, y_external,
                                     epochs, model_name,
                                     pretrain_model_path, batch_size, get_model_fun, *model_fun_arg):
    """cv_multi_input_one_fold"""
    X_tras = []
    X_vals = []
    for i in range(len(x_train)):
        X_tra, X_val, y_tra, y_val = train_test_split(x_train[i], y_train, train_size=train_ratio,
                                                      random_state=233,
                                                      stratify=evaluate_utils.one_hot_to_label(y_train))
        if x_external is not None:
            X_tra = np.vstack([x_external[i], X_tra])

        X_tras.append(X_tra)
        X_vals.append(X_val)
    if y_external is not None:
        y_tra = np.vstack((y_external, y_tra))
    model = get_model_fun(*model_fun_arg)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)

    y_pred = model.predict(X_vals, verbose=0)
    precision = evaluate_utils.precision(evaluate_utils.to_list(y_val),
                                         evaluate_utils.to_list(y_pred), 0.6)
    recall = evaluate_utils.recall(evaluate_utils.to_list(y_val),
                                   evaluate_utils.to_list(y_pred), 0.6)
    f1 = evaluate_utils.f1(evaluate_utils.to_list(y_val),
                           evaluate_utils.to_list(y_pred), 0.6)
    print("\n val - precision: %.6f - recall: %.6f - f1: %.6f \n"
          % (precision, recall, f1))

    f1 = F1(validation_data=(X_vals, y_val, X_tras, y_tra), interval=1)
    early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                   mode='max')
    best_model_filepath = model_path.model_file_dir + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                 save_best_only=True, save_weights_only=True,
                                 mode='max', period=1)
    # log_dir = model_path.model_log_dir + 'cv_multi_output_one_fold'
    # tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
    # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [f1, early_stopping, checkpoint]
    model.fit(X_tras, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
              callbacks=callbacks_list, class_weight='auto')
    print('best f1: %s' % str(checkpoint.best))
    model.load_weights(best_model_filepath)
    y_pred = model.predict(x_test, batch_size=1024)
    y_val_pred = None
    if x_val is not None:
        y_val_pred = model.predict(x_val, batch_size=1024)
    return y_pred, y_val_pred


class MultiLabelF1Backup(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(MultiLabelF1, self).__init__()

        self.interval = interval
        self.X_val, self.y_val, self.X_tra, self.y_tra = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            print('evaluate train')
            y_pred_tra = self.model.predict(self.X_tra, verbose=0)
            threshold = thresholds.topic_positive_threshold
            precision_tra = evaluate_utils.precision(evaluate_utils.to_normal_label(self.y_tra),
                                                     evaluate_utils.to_normal_label(y_pred_tra),
                                                     threshold)
            recall_tra = evaluate_utils.recall(evaluate_utils.to_normal_label(self.y_tra),
                                               evaluate_utils.to_normal_label(y_pred_tra), threshold)
            f1_tra = evaluate_utils.f1(evaluate_utils.to_normal_label(self.y_tra),
                                       evaluate_utils.to_normal_label(y_pred_tra), threshold)
            print("\n train - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                  % (epoch + 1, precision_tra, recall_tra, f1_tra))

            print('evaluate val')
            y_pred = self.model.predict(self.X_val, verbose=0)
            precision = evaluate_utils.precision(evaluate_utils.to_normal_label(self.y_val),
                                                 evaluate_utils.to_normal_label(y_pred), threshold)
            recall = evaluate_utils.recall(evaluate_utils.to_normal_label(self.y_val),
                                           evaluate_utils.to_normal_label(y_pred), threshold)
            f1 = evaluate_utils.f1(evaluate_utils.to_normal_label(self.y_val),
                                   evaluate_utils.to_normal_label(y_pred), threshold)
            print("\n val - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                  % (epoch + 1, precision, recall, f1))

            logs['f1'] = f1

            # 每个类别的验证精度
            for i in range(len(y_pred)):
                label = label_mapping.subject_mapping_reverse[str(i)]
                y_pred_i = evaluate_utils.to_list(y_pred[i])
                y_val_i = evaluate_utils.to_list(self.y_val[i])
                precision = evaluate_utils.precision(y_val_i, y_pred_i, [threshold[i]])
                recall = evaluate_utils.recall(y_val_i, y_pred_i, [threshold[i]])
                f1 = evaluate_utils.f1(y_val_i, y_pred_i, [threshold[i]])
                print("\n val - label_index: %d - label: %s - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                      % (i, label, epoch + 1, precision, recall, f1))


class MultiLabelF1OtherOutput(Callback):
    def __init__(self, validation_data=(), interval=1, num_class=10):
        super(MultiLabelF1OtherOutput, self).__init__()

        self.interval = interval
        self.num_class = num_class
        self.X_val, self.y_val, self.X_tra, self.y_tra = validation_data
        self.y_val = self.y_val[:self.num_class]
        self.y_tra = self.y_tra[:self.num_class]

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            print('evaluate train')
            y_pred_tra = self.model.predict(self.X_tra, verbose=0)[:self.num_class]
            threshold = thresholds.topic_positive_threshold
            precision_tra = evaluate_utils.precision(evaluate_utils.to_normal_label(self.y_tra),
                                                     evaluate_utils.to_normal_label(y_pred_tra),
                                                     threshold)
            recall_tra = evaluate_utils.recall(evaluate_utils.to_normal_label(self.y_tra),
                                               evaluate_utils.to_normal_label(y_pred_tra), threshold)
            f1_tra = evaluate_utils.f1(evaluate_utils.to_normal_label(self.y_tra),
                                       evaluate_utils.to_normal_label(y_pred_tra), threshold)
            print("\n train - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                  % (epoch + 1, precision_tra, recall_tra, f1_tra))

            print('evaluate val')
            y_pred = self.model.predict(self.X_val, verbose=0)[:self.num_class]
            precision = evaluate_utils.precision(evaluate_utils.to_normal_label(self.y_val),
                                                 evaluate_utils.to_normal_label(y_pred), threshold)
            recall = evaluate_utils.recall(evaluate_utils.to_normal_label(self.y_val),
                                           evaluate_utils.to_normal_label(y_pred), threshold)
            f1 = evaluate_utils.f1(evaluate_utils.to_normal_label(self.y_val),
                                   evaluate_utils.to_normal_label(y_pred), threshold)
            print("\n val - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                  % (epoch + 1, precision, recall, f1))

            logs['f1'] = f1

            # 每个类别的验证精度
            for i in range(len(y_pred)):
                label = label_mapping.subject_mapping_reverse[str(i)]
                y_pred_i = evaluate_utils.to_list(y_pred[i])
                y_val_i = evaluate_utils.to_list(self.y_val[i])
                precision = evaluate_utils.precision(y_val_i, y_pred_i, [threshold[i]])
                recall = evaluate_utils.recall(y_val_i, y_pred_i, [threshold[i]])
                f1 = evaluate_utils.f1(y_val_i, y_pred_i, [threshold[i]])
                print("\n val - label_index: %d - label: %s - epoch: %d - precision: %.6f - recall: %.6f - f1: %.6f \n"
                      % (i, label, epoch + 1, precision, recall, f1))


def get_best_thresholds(y_val_true, y_val_pred):
    result = []
    for i in range(len(y_val_pred)):
        best_f1 = 0
        this_best_threshold = 0.7
        threshold = 0.2
        while threshold < 0.8:
            y_val_pred_i = evaluate_utils.to_list(y_val_pred[i])
            y_val_true_i = evaluate_utils.to_list(y_val_true[i])
            f1 = evaluate_utils.f1(y_val_true_i, y_val_pred_i, [threshold])
            if f1 > best_f1:
                this_best_threshold = threshold
                best_f1 = f1
            threshold += 0.02
        result.append(this_best_threshold)
        print('label_index: %d best_threshold: %f best_f1:%f' % (i, this_best_threshold, best_f1))

    precision = evaluate_utils.precision(evaluate_utils.to_normal_label(y_val_true),
                                         evaluate_utils.to_normal_label(y_val_pred), result)
    recall = evaluate_utils.recall(evaluate_utils.to_normal_label(y_val_true),
                                   evaluate_utils.to_normal_label(y_val_pred), result)
    f1 = evaluate_utils.f1(evaluate_utils.to_normal_label(y_val_true),
                           evaluate_utils.to_normal_label(y_val_pred), result)
    print("\n train - precision: %.6f - recall: %.6f - f1: %.6f \n" % (precision, recall, f1))
    return result


def cv_multi_output_one_fold(x_train, y_train, train_ratio, x_test, x_val, epochs, model_name,
                             pretrain_model_path, get_model_fun, *model_fun_arg):
    """cv_multi_output_one_fold"""
    X_tra, X_val = train_test_split(x_train, train_size=train_ratio, random_state=233)
    y_tras = []
    y_vals = []
    for i in range(len(y_train)):
        y_tra, y_val = train_test_split(y_train[i], train_size=train_ratio,random_state=233)
        y_tras.append(y_tra)
        y_vals.append(y_val)

    batch_size = 16
    model = get_model_fun(*model_fun_arg)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)

    f1 = MultiLabelF1(validation_data=(X_val, y_vals, X_tra, y_tras), interval=1)
    early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                   mode='max')
    best_model_filepath = model_path.model_file_dir + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0, save_best_only=True,
                                 save_weights_only=True, mode='max',
                    period=1)
    # log_dir = model_path.model_log_dir + 'cv_multi_output_one_fold'
    # tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
    # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [f1, early_stopping, checkpoint]
    model.fit(X_tra, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(X_val, y_vals),
              callbacks=callbacks_list)
    print('best f1: %s' % str(checkpoint.best))
    model.load_weights(best_model_filepath)

    best_thresholds = get_best_thresholds(model, X_val, y_vals)
    thresholds.topic_positive_threshold = best_thresholds
    print('thresholds.topic_positive_threshold:')
    print(thresholds.topic_positive_threshold)

    y_pred = model.predict(x_test, batch_size=1)
    y_val = None
    if x_val is not None:
        y_val = model.predict(x_val, batch_size=1)
        y_val = evaluate_utils.to_normal_label_ndarray(y_val)
    return evaluate_utils.to_normal_label_ndarray(y_pred), y_val


def cv_multi_output_one_fold_external(x_train, y_train, train_ratio, x_test, x_val, epochs, model_name,
                                      pretrain_model_path, x_external, y_external, get_model_fun, *model_fun_arg):
    """cv_multi_output_one_fold"""
    X_tra, X_val = train_test_split(x_train, train_size=train_ratio, random_state=233)
    if x_external is not None:
        X_tra = np.vstack([X_tra, x_external])
    y_tras = []
    y_vals = []
    for i in range(len(y_train)):
        y_tra, y_val = train_test_split(y_train[i], train_size=train_ratio,random_state=233)
        if y_external is not None:
            y_tra = np.vstack((y_tra, y_external[i]))
        y_tras.append(y_tra)
        y_vals.append(y_val)

    batch_size = 16
    model = get_model_fun(*model_fun_arg)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)

    f1 = MultiLabelF1(validation_data=(X_val, y_vals, X_tra, y_tras), interval=1)
    early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                   mode='max')
    best_model_filepath = model_path.model_file_dir + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0, save_best_only=True,
                                 save_weights_only=True, mode='max',
                    period=1)
    # log_dir = model_path.model_log_dir + 'cv_multi_output_one_fold'
    # tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
    # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [f1, early_stopping, checkpoint]
    model.fit(X_tra, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(X_val, y_vals),
              callbacks=callbacks_list)

    print('best f1: %s' % str(checkpoint.best))

    model.load_weights(best_model_filepath)

    y_pred = model.predict(x_test, batch_size=1)
    y_val = None
    if x_val is not None:
        y_val = model.predict(x_val, batch_size=1)
        y_val = evaluate_utils.to_normal_label_ndarray(y_val)
    return evaluate_utils.to_normal_label_ndarray(y_pred), y_val


def cv_multi_input_multi_output_one_fold(x_train, y_train, train_ratio, x_test, epochs, model_name,
                                         get_model_fun, *model_fun_arg):
    """cv_multi_input_multi_output_one_fold"""
    X_tras = []
    X_vals = []
    y_tras = []
    y_vals = []
    for i in range(len(x_train)):
        X_tra, X_val = train_test_split(x_train[i], train_size=train_ratio,random_state=233)
        X_tras.append(X_tra)
        X_vals.append(X_val)
    for i in range(len(y_train)):
        y_tra, y_val = train_test_split(y_train[i], train_size=train_ratio,random_state=233)
        y_tras.append(y_tra)
        y_vals.append(y_val)

    batch_size = 16
    model = get_model_fun(*model_fun_arg)
    f1 = MultiLabelF1(validation_data=(X_vals, y_vals, X_tras, y_tras), interval=1)
    early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                   mode='max')
    best_model_filepath = model_path.model_file_dir + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0, save_best_only=True,
                                 save_weights_only=True, mode='max', period=1)
    # log_dir = model_path.model_log_dir + 'cv_multi_output_one_fold'
    # tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
    # embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [f1, early_stopping, checkpoint]
    model.fit(X_tras, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(X_vals, y_vals), callbacks=callbacks_list)
    print('best f1: %s' % str(checkpoint.best))
    model.load_weights(best_model_filepath)
    y_pred = model.predict(x_test, batch_size=1024)
    return y_pred


def cv_multi_input_multi_output(x_train, y_train, k, x_test, x_val, epochs, monitor_threshold, model_name,
                                pretrain_model_path, batch_size, is_one_fold, sample_weight, num_class,
                                get_model_fun, *model_fun_arg):
    """cv_multi_input_multi_output"""
    result = []
    result_val = []
    ntrain = x_train[0].shape[0]
    oof_train = np.zeros((ntrain, num_class))
    kf = KFold(n_splits=k, shuffle=True, random_state=233)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train[0], y_train[0]):
        X_tras = []
        X_vals = []
        y_tras = []
        y_vals = []
        for i in range(len(x_train)):
            X_tra, X_val = x_train[i][train_index], x_train[i][test_index]
            X_tras.append(X_tra)
            X_vals.append(X_val)
        for i in range(len(y_train)):
            y_tra, y_val = y_train[i][train_index], y_train[i][test_index]
            y_tras.append(y_tra)
            y_vals.append(y_val)

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = MultiLabelF1(validation_data=(X_vals, y_vals, X_tras, y_tras), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=2, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + str(serial_num) + '.hdf5'
        serial_num += 1
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True,
                                     save_weights_only=True, mode='max',
                                     period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        model.fit(X_tras, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_vals, y_vals),
                  callbacks=callbacks_list)
        bests.append(checkpoint.best)
        if checkpoint.best < monitor_threshold:
            continue
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        y_pred = evaluate_utils.to_normal_label_ndarray(y_pred)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_val is not None:
            y_val = model.predict(x_val, batch_size=1)
            y_val = evaluate_utils.to_normal_label_ndarray(y_val)
            result_val.append(y_val)

        oof_train[test_index] = evaluate_utils.to_normal_label_ndarray(model.predict(X_vals))
        serial_num += 1
        if is_one_fold:
            break

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    y_test = average_of_pred(result)
    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, average_of_pred(result_val)


def average_of_pred(preds):
    if preds is None or len(preds) == 0:
        return None
    result = copy.deepcopy(preds[0])
    for i in range(1, len(preds)):
        result += preds[i]
    result /= len(preds)
    return result


def average_of_prediction_of_muliti_out(preds):
    if preds is None or len(preds) == 0:
        return None
    result = []
    for i in range(len(preds[0])):
        pred_i = []
        for j in range(len(preds)):
            pred_i.append(preds[j][i])
        result.append(average_of_pred(pred_i))
    return result


def cv_multi_output(x_train, y_train, k, x_test, x_val, epochs, monitor_threshold, model_name,
                    pretrain_model_path, is_one_fold, batch_size, sample_weight, class_num, get_model_fun, *model_fun_arg):
    """cv_multi_output"""
    result = []
    result_val = []
    ntrain = x_train.shape[0]
    oof_train = np.zeros((ntrain, class_num))
    kf = KFold(n_splits=k, shuffle=True, random_state=233)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train, y_train[0]):
        y_tras = []
        y_vals = []
        X_tra, X_val = x_train[train_index], x_train[test_index]
        for i in range(len(y_train)):
            y_tra, y_val = y_train[i][train_index], y_train[i][test_index]
            y_tras.append(y_tra)
            y_vals.append(y_val)

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = MultiLabelF1(validation_data=(X_val, y_vals, X_tra, y_tras), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=6, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        if sample_weight is not None:
            sample_weight_temp = [weight[train_index] for weight in sample_weight]
        else:
            sample_weight_temp = None
        model.fit(X_tra, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_vals),
                  callbacks=callbacks_list, sample_weight=sample_weight_temp)
        bests.append(checkpoint.best)
        print('best f1: %s' % str(checkpoint.best))
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        y_pred = evaluate_utils.to_normal_label_ndarray(y_pred)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_val is not None:
            y_val = model.predict(x_val, batch_size=1)
            y_val = evaluate_utils.to_normal_label_ndarray(y_val)
            result_val.append(y_val)

        oof_train[test_index] = evaluate_utils.to_normal_label_ndarray(model.predict(X_val))
        serial_num += 1
        if is_one_fold:
            break

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    y_test = average_of_pred(result)
    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, average_of_pred(result_val)


def cv_multi_output_dev(x_train, y_train, k, x_test, x_dev, y_dev, epochs, monitor_threshold, model_name,
                    pretrain_model_path, is_one_fold, batch_size, sample_weight, get_model_fun, *model_fun_arg):
    """cv_multi_output"""
    result = []
    result_val = []
    ntrain = x_train.shape[0]
    oof_train = np.zeros((ntrain, 10))
    kf = KFold(n_splits=k, shuffle=True, random_state=233)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train, y_train[0]):
        y_tras = []
        y_vals = []
        X_tra, X_val = x_train[train_index], x_train[test_index]
        for i in range(len(y_train)):
            y_tra, y_val = y_train[i][train_index], y_train[i][test_index]
            y_tras.append(y_tra)
            y_vals.append(y_val)

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = MultiLabelF1(validation_data=(X_val, y_vals, X_tra, y_tras), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=6, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        if sample_weight is not None:
            sample_weight_temp = [weight[train_index] for weight in sample_weight]
        else:
            sample_weight_temp = None
        model.fit(X_tra, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(x_dev, y_dev),
                  callbacks=callbacks_list, sample_weight=sample_weight_temp)
        bests.append(checkpoint.best)
        print('best f1: %s' % str(checkpoint.best))
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        y_pred = evaluate_utils.to_normal_label_ndarray(y_pred)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_dev is not None:
            y_val = model.predict(x_dev, batch_size=1)
            y_val = evaluate_utils.to_normal_label_ndarray(y_val)
            result_val.append(y_val)

        oof_train[test_index] = evaluate_utils.to_normal_label_ndarray(model.predict(X_val))
        serial_num += 1
        if is_one_fold:
            break

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    y_test = average_of_pred(result)
    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, average_of_pred(result_val)


def cv_multi_output_count(x_train, y_train, k, x_test, x_val, epochs, monitor_threshold, model_name,
                    pretrain_model_path, is_one_fold, batch_size, sample_weight, get_model_fun, *model_fun_arg):
    """cv_multi_output"""
    result = []
    result_val = []
    ntrain = x_train.shape[0]
    oof_train = np.zeros((ntrain, 10))
    kf = KFold(n_splits=k, shuffle=True, random_state=233)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train, y_train[0]):
        y_tras = []
        y_vals = []
        X_tra, X_val = x_train[train_index], x_train[test_index]
        for i in range(len(y_train)):
            y_tra, y_val = y_train[i][train_index], y_train[i][test_index]
            y_tras.append(y_tra)
            y_vals.append(y_val)

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = MultiLabelF1OtherOutput(validation_data=(X_val, y_vals, X_tra, y_tras), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=6, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        if sample_weight is not None:
            sample_weight_temp = [weight[train_index] for weight in sample_weight]
        else:
            sample_weight_temp = None
        model.fit(X_tra, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_vals),
                  callbacks=callbacks_list, sample_weight=sample_weight_temp)
        bests.append(checkpoint.best)
        print('best f1: %s' % str(checkpoint.best))
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)[:10]
        y_pred = evaluate_utils.to_normal_label_ndarray(y_pred)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_val is not None:
            y_val = model.predict(x_val, batch_size=1)[:10]
            y_val = evaluate_utils.to_normal_label_ndarray(y_val)
            result_val.append(y_val)

        oof_train[test_index] = evaluate_utils.to_normal_label_ndarray(model.predict(X_val)[:10])
        serial_num += 1
        if is_one_fold:
            break

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    y_test = average_of_pred(result)
    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, average_of_pred(result_val)


def bagging_multi_output(x_train, y_train, k, x_test, epochs, batch_size, model_names,
                    models, *model_fun_arg):
    """
    需要实现:
    1. 支持不同模型
    """
    result = []
    kf = KFold(n_splits=k, shuffle=True, random_state=233)
    serial_num = 0
    bests = []
    file_utils.write_lines([''], model_path.bagging_multi_label_topic_file_path)
    for train_index, test_index in kf.split(x_train, y_train[0]):
        for i, model in enumerate(models):
            np.random.seed(i + 42)
            tf.set_random_seed(i + 2)
            X_tra, X_val = x_train[train_index], x_train[test_index]
            y_tras = []
            y_vals = []
            for j in range(len(y_train)):
                y_tra, y_val = y_train[j][train_index], y_train[j][test_index]
                y_tras.append(y_tra)
                y_vals.append(y_val)

            model = model(*model_fun_arg)
            f1 = MultiLabelF1(validation_data=(X_val, y_vals, X_tra, y_tras), interval=1)
            early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=10, verbose=0,
                                           mode='max')
            best_model_filepath = model_path.model_file_dir + model_names[i] + '_' + str(serial_num) + '.hdf5'
            serial_num += 1
            file_utils.write_lines(['best_model_filepath'], model_path.bagging_multi_label_topic_file_path, mode='w')
            checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                         save_best_only=True, save_weights_only=True,
                                         mode='max', period=1)
            callbacks_list = [f1, early_stopping, checkpoint]
            model.fit(X_tra, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
                      validation_data=(X_val, y_vals),
                      callbacks=callbacks_list)
            bests.append(checkpoint.best)
            print('best f1: %s' % str(checkpoint.best))
            model.load_weights(best_model_filepath)
            y_pred = model.predict(x_test, batch_size=1024)
            result.append(evaluate_utils.to_normal_label_ndarray(y_pred))

    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    y_test = result[0]
    for i in range(1, len(result)):
        y_test += result[i]
    y_test /= len(result)
    return y_test


def cv_multi_output_external(x_train, y_train, k, x_test, epochs, monitor_threshold, model_name,
                             pretrain_model_path, x_external, y_external,
                             get_model_fun, *model_fun_arg):
    """cv_multi_output"""
    result = []
    # ntrain = x_train.shape[0]
    # oof_train = np.zeros((ntrain, class_num))
    kf = KFold(n_splits=k, shuffle=True, random_state=233)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train, y_train[0]):
        y_tras = []
        y_vals = []
        X_tra, X_val = x_train[train_index], x_train[test_index]
        if x_external is not None:
            X_tra = np.vstack([X_tra, x_external])
        for i in range(len(y_train)):
            y_tra, y_val = y_train[i][train_index], y_train[i][test_index]
            if y_external is not None:
                y_tra = np.vstack((y_tra, y_external[i]))
            y_tras.append(y_tra)
            y_vals.append(y_val)

        batch_size = 16
        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = MultiLabelF1(validation_data=(X_val, y_vals, X_tra, y_tras), interval=1)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=4, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + str(serial_num) + '.hdf5'
        serial_num += 1
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        model.fit(X_tra, y_tras, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_vals),
                  callbacks=callbacks_list)
        bests.append(checkpoint.best)
        print('best f1: %s' % str(checkpoint.best))
        if checkpoint.best < monitor_threshold:
            continue
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        result.append(evaluate_utils.to_normal_label_ndarray(y_pred))

    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    y_test = result[0]
    for i in range(1, len(result)):
        y_test += result[i]
    y_test /= len(result)
    return y_test


def cv(x_train, y_train, k, x_test, x_val,epochs, monitor_threshold, model_name, pretrain_model_path,
       batch_size, is_one_fold, sample_weight, num_class, get_model_fun, *model_fun_arg):
    """cv_multi_output"""
    result = []
    result_val = []
    ntrain = x_train.shape[0]
    oof_train = np.zeros((ntrain, num_class))
    kf = KFold(n_splits=k, shuffle=True, random_state=233)
    serial_num = 0
    bests = []
    for train_index, test_index in kf.split(x_train, y_train):
        X_tra, X_val = x_train[train_index], x_train[test_index]
        y_tra, y_val = y_train[train_index], y_train[test_index]

        model = get_model_fun(*model_fun_arg)
        if pretrain_model_path:
            model.load_weights(pretrain_model_path)
        f1 = F1(validation_data=(X_val, y_val, X_tra, y_tra), interval=1, num_class=num_class)
        early_stopping = EarlyStopping(monitor='f1', min_delta=0.00001, patience=10, verbose=0,
                                       mode='max')
        best_model_filepath = model_path.model_file_dir + model_name + '_' + str(
            serial_num) + '.hdf5'
        checkpoint = ModelCheckpoint(best_model_filepath, monitor='f1', verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='max', period=1)
        callbacks_list = [f1, early_stopping, checkpoint]
        if sample_weight is not None:
            sample_weight_temp = sample_weight[train_index]
        else:
            sample_weight_temp = None
        model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, verbose=2,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks_list, sample_weight=sample_weight_temp)
        print('best f1: %s' % str(checkpoint.best))
        if checkpoint.best < monitor_threshold:
            continue
        bests.append(checkpoint.best)
        model.load_weights(best_model_filepath)
        y_pred = model.predict(x_test, batch_size=1024)
        np.save(data_path.data_base_dir + model_name + '_' + str(serial_num) + '.test', y_pred)
        result.append(y_pred)

        if x_val is not None:
            y_val = model.predict(x_val, batch_size=1)
            result_val.append(y_val)

        oof_train[test_index] = model.predict(X_val)
        serial_num += 1
        if is_one_fold:
            break

    np.save(data_path.data_base_dir + model_name + '.train', oof_train)
    average_of_best = sum(bests) / len(bests)
    print('average_of_best: %f' % average_of_best)

    y_test = average_of_pred(result)

    np.save(data_path.data_base_dir + model_name + '.test', y_test)

    return y_test, average_of_pred(result_val)
