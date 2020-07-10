#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tf 1.14"""

__author__ = 'yp'

import os
import time
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from ner_model_4 import Model4 as Model
from sklearn.metrics import classification_report
from data_process import DataProcess
from CRFSuiteForNER import SentenceGetter
from sklearn.model_selection import train_test_split
from mylogparse import *

max_sentence_len = 1500

a = LogParse()
a.set_profile(path="./", filename="exp")

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

lll = list(label_dict.values())
lll.remove('O')

model_dir = os.path.join('./', '_'.join([Model.__name__, time.strftime("%Y%m%d%H%M%S")]))
if os.path.exists(model_dir):
    pass
else:
    os.mkdir(model_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURE_MODE = 'pre_train'
TRAIN_MODE = 'train'

df = pd.read_csv('crf_train.txt', quoting=csv.QUOTE_NONE,
                 encoding="utf-8", sep='\t', header=None)
df.columns = ['Sentence #', 'word', 'tag']
df = df.fillna(method='ffill')

getter = SentenceGetter(df)
sentences = getter.sentences

train_sentences, test_sentences = train_test_split(sentences, test_size=0.05, random_state=0)

model = Model(learning_rate=0.0001, sequence_length_val=max_sentence_len, num_tags=13)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

log_file = './log_file.txt'

if TRAIN_MODE == 'train':
    with open(log_file, mode='w', encoding='utf-8') as f1:
        with tf.Session(config=config) as sess:
            sess.run(init)

            train_data_process = DataProcess(train_sentences, max_length=max_sentence_len)
            train_data_process.get_feature()

            test_data_process = DataProcess(test_sentences, max_length=max_sentence_len)
            test_data_process.get_feature()

            step = 0
            epoch = 40

            for i in range(epoch):

                for _, batch_x, batch_y in train_data_process.next_batch():
                    sum_counter = 0
                    right_counter = 0

                    model.is_training = True
                    _seq_len = np.array([len(_) for _ in batch_x])
                    _logits, _loss, _opt, transition_params = sess.run([model.logits,
                                                                        model.loss_val,
                                                                        model.train_op,
                                                                        model.transition_params
                                                                        ],
                                                                       feed_dict={model.input_x: batch_x,
                                                                                  model.input_y: batch_y,
                                                                                  model.sequence_lengths: _seq_len,
                                                                                  model.keep_prob: 0.8})

                    step += 1

                    for logit, seq_len, _y_label in zip(_logits, _seq_len, batch_y):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                        for tmp_seq, tmp_label in zip(viterbi_seq, _y_label):

                            if tmp_seq == tmp_label:
                                right_counter += 1
                            sum_counter += 1

                    if step % 10 == 0:
                        print("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
                        a.info("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
                        f1.writelines("step:{} ===loss:{} === acc: {}\n".format(str(step),
                                                                                str(_loss),
                                                                                str(right_counter / sum_counter)))

                save_path = saver.save(sess, "%s/%s/model_epoch_%s" % (model_dir, str(i), str(i)))

                # test
                y_predict_list = []
                y_label_list = []

                sum_counter = 0
                right_counter = 0
                f1_statics = np.array([0 for i in range(12)])
                y_t = []
                y_p = []
                for batch_sentence, batch_x, batch_y in test_data_process.next_batch():
                    model.is_training = False
                    _seq_len = np.array([len(_) for _ in batch_x])
                    _logits, transition_params = sess.run([model.logits,
                                                           model.transition_params], feed_dict=
                                                          {model.input_x: batch_x,
                                                           model.input_y: batch_y,
                                                           model.sequence_lengths: _seq_len,
                                                           model.keep_prob: 1.0})
                    for _sentence_str, logit, seq_len, _y_label in zip(batch_sentence, _logits, _seq_len, batch_y):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                        y_p.extend(viterbi_seq)
                        y_t.extend(list(_y_label))
                _ = classification_report([label_map_reverse[__] for __ in y_t],
                                          [label_map_reverse[__] for __ in y_p],
                                          labels=lll)
                print(_)
                a.info("epoch: {}====classification_report: {} \n".format(str(i), str(_)))
                f1.writelines("step:{} === classification_report:{}\n".format(str(i), str(_)))

if TRAIN_MODE == 'predict':
    predict_data_list = ['../data/medical_record/train_3w.txt']

    predict_data_process = DataProcess(feature_mode=FEATURE_MODE)
    predict_data_process.load_data(file_list=predict_data_list)
    predict_data_process.get_feature()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../model/19/model_epoch_19")

        y_predict_list = []
        for batch_x, batch_y in predict_data_process.next_batch():
            model.is_training = False
            _seq_len = np.array([len(_) for _ in batch_x])
            _logits, transition_params = sess.run([model.logits,
                                                   model.transition_params],
                                                  feed_dict={model.input_x: batch_x,
                                                             model.sequence_lengths: _seq_len,
                                                             model.keep_prob: 1.0})

            for logit, seq_len in zip(_logits, _seq_len):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                y_predict_list.append(viterbi_seq)

        _out_file = predict_data_process.data
        _out_file['y_pred'] = pd.Series(y_predict_list)
        _out_file.to_csv('./final_predict.tsv', sep='\t')
