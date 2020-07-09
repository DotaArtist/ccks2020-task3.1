#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from ner_model_1 import Model1 as Model
from sklearn.metrics import classification_report
from data_process import DataProcess


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FEATURE_MODE = 'pre_train'
TRAIN_MODE = 'train'

model_dir = os.path.join('../', '_'.join([Model.__name__, time.strftime("%Y%m%d%H%M%S")]))
a.info('model_dir:==={}'.format(model_dir))
print('model_dir:==={}'.format(model_dir))

if os.path.exists(model_dir):
    pass
else:
    os.mkdir(model_dir)

log_file = './log_file.txt'
_num = 0.00001

train_data_list = [
    # '../data/normal_train/train_v3.txt',
    # '../data/normal_train/ner_2w_checked.txt',
    # '../data/normal_train/ocr_1_resample_60.txt',
    # '../data/normal_train/ccks_train.txt',
    # '../data/normal_train/ccks_2019_wu_null.txt',
    # '../data/normal_train/train_v3_wu_null.txt',
    # '../data/normal_train/ccks_2019_wu_notnull_new.txt',
    # '../data/normal_train/add_deny_resample_3.txt',
    # '../data/normal_train/v2_0815.txt',
    '../data/normal_train/test_v2.txt'
]

test_data_list = ['../data/normal_train/test_v2.txt']

model = Model(learning_rate=0.0001, sequence_length_val=max_sentence_len, num_tags=9)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if TRAIN_MODE == 'train':
    with open(log_file, mode='w', encoding='utf-8') as f1:
        with tf.Session(config=config) as sess:
            sess.run(init)

            train_data_process = DataProcess(feature_mode=FEATURE_MODE)
            train_data_process.load_data(file_list=train_data_list)
            train_data_process.get_feature()

            test_data_process = DataProcess(feature_mode=FEATURE_MODE)
            test_data_process.load_data(file_list=test_data_list, is_shuffle=False)
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

                        if viterbi_seq == list(_y_label):
                            right_counter += 1
                        sum_counter += 1

                    if step % 50 == 0:
                        a.info("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
                        print("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
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
