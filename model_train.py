#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.Generator = None

import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from CRFSuiteForNER import SentenceGetter
from seqeval.callbacks import F1Metrics
from seqeval.metrics import classification_report
from tensorflow_addons.text.crf import viterbi_decode
from ner_model_1 import Model1 as Model
from sklearn.metrics import classification_report
from data_process import DataProcess
from CRF import CRF
import keras
from keras.callbacks import ModelCheckpoint

label_dict = {
    'B-疾病和诊断': 'B-disease',
    'B-影像检查': 'B-check',
    'B-解剖部位': 'B-body',
    'B-手术': 'B-operation',
    'B-药物': 'B-drug',
    'B-实验室检验': 'B-analysis',
    'I-疾病和诊断': 'I-disease',
    'I-影像检查': 'I-check',
    'I-解剖部位': 'I-body',
    'I-手术': 'I-operation',
    'I-药物': 'I-drug',
    'I-实验室检验': 'I-analysis',
    'O-O': 'O',
}

label_map = {
    'B-disease': 1,
    'I-disease': 2,
    'B-check': 3,
    'I-check': 4,
    'B-body': 5,
    'I-body': 6,
    'B-operation': 7,
    'I-operation': 8,
    'B-drug': 9,
    'I-drug': 10,
    'B-analysis': 11,
    'I-analysis': 12,
    'O': 0,
}

label_map_reverse = {
    1: 'B-disease',
    2: 'I-disease',
    3: 'B-check',
    4: 'I-check',
    5: 'B-body',
    6: 'I-body',
    7: 'B-operation',
    8: 'I-operation',
    9: 'B-drug',
    10: 'I-drug',
    11: 'B-analysis',
    12: 'I-analysis',
    0: 'O'
}


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.f1 = []
        self._data = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURE_MODE = 'pre_train'
TRAIN_MODE = 'train'

df = pd.read_csv('crf_train.txt',
                 encoding="utf-8", sep='\t', header=None)
df.columns = ['Sentence #', 'word', 'tag']
df = df.fillna(method='ffill')

getter = SentenceGetter(df)
sentences = getter.sentences

train_sentences, test_sentences = train_test_split(sentences, test_size=0.05, random_state=0)
train_data_process = DataProcess(train_sentences)
train_data_process.get_feature()

test_data_process = DataProcess(test_sentences)
test_data_process.get_feature()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(13, return_sequences=True, activation="tanh"), merge_mode='sum'))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(13, return_sequences=True, activation="softmax"), merge_mode='sum'))
crf = CRF(13, name='crf_layer')
model.add(crf)
model.compile('adam', loss=crf.get_loss)

history = LossHistory()
checkpointer = ModelCheckpoint(filepath='./model/weights.hdf5', verbose=1, save_best_only=True)
model.fit(train_data_process.data_x, train_data_process.data_y,
          batch_size=32, epochs=200,
          validation_data=(test_data_process.data_x, test_data_process.data_y),
          # callbacks=[history, checkpointer, F1Metrics(label_map_reverse)],
          callbacks=[F1Metrics(label_map_reverse)])

# if TRAIN_MODE == 'train':
#     with tf.Session(config=config) as sess:
#         sess.run(init)
#
#         train_data_process = DataProcess(train_sentences)
#         train_data_process.get_feature()
#
#         test_data_process = DataProcess(test_sentences)
#         test_data_process.get_feature()
#
#         step = 0
#         epoch = 40
#
#         for i in range(epoch):
#
#             for _, batch_x, batch_y in train_data_process.next_batch():
#                 sum_counter = 0
#                 right_counter = 0
#
#                 model.is_training = True
#                 _seq_len = np.array([len(_) for _ in batch_x])
#                 _logits, _loss, _opt, transition_params = sess.run([model.logits,
#                                                                     model.loss_val,
#                                                                     model.train_op,
#                                                                     model.transition_params
#                                                                     ],
#                                                                    feed_dict={model.input_x: batch_x,
#                                                                               model.input_y: batch_y,
#                                                                               model.sequence_lengths: _seq_len,
#                                                                               model.keep_prob: 0.8})
#
#                 step += 1
#
#                 for logit, seq_len, _y_label in zip(_logits, _seq_len, batch_y):
#                     viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
#
#                     if viterbi_seq == list(_y_label):
#                         right_counter += 1
#                     sum_counter += 1
#
#                 if step % 50 == 0:
#                     print("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
#                     f1.writelines("step:{} ===loss:{} === acc: {}\n".format(str(step),
#                                                                             str(_loss),
#                                                                             str(right_counter / sum_counter)))
#
#             save_path = saver.save(sess, "%s/%s/model_epoch_%s" % (model_dir, str(i), str(i)))
#
#             # test
#             y_predict_list = []
#             y_label_list = []
#
#             sum_counter = 0
#             right_counter = 0
#             f1_statics = np.array([0 for i in range(12)])
#             y_t = []
#             y_p = []
#             for batch_sentence, batch_x, batch_y in test_data_process.next_batch():
#                 model.is_training = False
#                 _seq_len = np.array([len(_) for _ in batch_x])
#                 _logits, transition_params = sess.run([model.logits,
#                                                        model.transition_params], feed_dict=
#                                                       {model.input_x: batch_x,
#                                                        model.input_y: batch_y,
#                                                        model.sequence_lengths: _seq_len,
#                                                        model.keep_prob: 1.0})
#                 for _sentence_str, logit, seq_len, _y_label in zip(batch_sentence, _logits, _seq_len, batch_y):
#                     viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
#
#                     y_p.extend(viterbi_seq)
#                     y_t.extend(list(_y_label))
