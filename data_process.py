#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据读取"""

__author__ = 'yp'

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

embedding_path = "./medical_record_character_embedding.txt"
max_sentence_len = 128

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


class PreTrainProcess(object):
    def __init__(self, path=embedding_path, embedding_dim=256,
                 sentence_len=max_sentence_len, pair_mode=False):
        embeddings = dict()

        self.embedding_path = path
        self.embedding_dim = embedding_dim
        self.sentence_len = sentence_len
        self.pair_mode = pair_mode

        with open(self.embedding_path, encoding='utf-8', mode='r') as f1:
            for line in f1.readlines():
                line = line.strip().split(' ')
                character = line[0]
                vector = [float(i) for i in line[1:]]

                if character not in embeddings.keys():
                    embeddings[character] = vector
        print('pre train feature loaded.')
        self.embedding_dict = embeddings

    def encode(self, sentence, **kwargs):
        if 'pair_mode' in kwargs.keys():
            if not isinstance(kwargs['pair_mode'], bool):
                raise TypeError("mode type must bool!")

        if 'pair_mode' in kwargs.keys() and kwargs['pair_mode']:
            try:
                assert isinstance(sentence, list)
            except AssertionError:
                print("sentence must be list!")
        else:
            try:
                assert isinstance(sentence, list)
                embedding_unk = [0.0 for _ in range(self.embedding_dim)]
                out_put = []

                for sentence_idx, _sentence in enumerate(sentence):
                    out_put_tmp = []

                    for char_idx, _char in enumerate(list(_sentence)):
                        if char_idx < self.sentence_len:
                            out_put_tmp.append(self.embedding_dict.get(_char, embedding_unk))

                    for i in range(self.sentence_len - len(out_put_tmp)):
                        out_put_tmp.append(embedding_unk)

                    out_put_tmp = np.stack(out_put_tmp, axis=0)
                    out_put.append(out_put_tmp)

                return np.stack(out_put, axis=0)
            except AssertionError:
                print("sentence must be list!")


class PreTrainElmoProcess(object):
    def __init__(self, path=embedding_path, embedding_dim=512,
                 sentence_len=max_sentence_len, pair_mode=False):
        embeddings = dict()

        self.embedding_path = path
        self.embedding_dim = embedding_dim
        self.sentence_len = sentence_len
        self.pair_mode = pair_mode
        self.embedding_dict = embeddings

        g_elmo = tf.Graph()
        vocab_file = './bilmelmo/data/vocab.txt'
        options_file = './bilmelmo/try/options.json'
        weight_file = './bilmelmo/try/weights.hdf5'
        token_embedding_file = './bilmelmo/data/vocab_embedding.hdf5'

        with tf.Graph().as_default() as g_elmo:
            self.batcher = TokenBatcher(vocab_file)
            self.context_token_ids = tf.placeholder('int32', shape=(None, None))
            self.bilm = BidirectionalLanguageModel(
                options_file,
                weight_file,
                use_character_inputs=False,
                embedding_weight_file=token_embedding_file
            )

            self.context_embeddings_op = self.bilm(self.context_token_ids)
            self.elmo_context_input = weight_layers('input', self.context_embeddings_op, l2_coef=0.0)

            self.elmo_context_output = weight_layers(
                'output', self.context_embeddings_op, l2_coef=0.0
            )
            init = tf.global_variables_initializer()
        sess_elmo = tf.Session(graph=g_elmo)
        sess_elmo.run(init)
        self.sess_elmo = sess_elmo

    def encode(self, sentence, **kwargs):
        if 'pair_mode' in kwargs.keys():
            if not isinstance(kwargs['pair_mode'], bool):
                raise TypeError("mode type must bool!")

        if 'pair_mode' in kwargs.keys() and kwargs['pair_mode']:
            try:
                assert isinstance(sentence, list)
            except AssertionError:
                print("sentence must be list!")
        else:
            try:
                assert isinstance(sentence, list)
                embedding_unk = [0.0 for _ in range(self.embedding_dim)]
                out_put = []

                for sentence_idx, _sentence in enumerate(sentence):

                    context_ids = self.batcher.batch_sentences(list(_sentence))
                    out_put_tmp = self.sess_elmo.run(
                        [self.elmo_context_input['weighted_op']],
                        feed_dict={self.context_token_ids: context_ids}
                    )[0][0].tolist()

                    for i in range(self.sentence_len - len(out_put_tmp)):
                        out_put_tmp.append(embedding_unk)

                    out_put_tmp = np.stack(out_put_tmp, axis=0)
                    out_put.append(out_put_tmp)

                return np.stack(out_put, axis=0)
            except AssertionError:
                print("sentence must be list!")


class EmbeddingPreTrain(object):
    def __init__(self):
        # self.model = PreTrainProcess(path=embedding_path, embedding_dim=256,
        #                              sentence_len=max_sentence_len)
        self.model = PreTrainElmoProcess(path=embedding_path, embedding_dim=512,
                                     sentence_len=max_sentence_len)

    def get_output(self, sentence, _show_tokens=True):
        try:
            return self.model.encode(sentence, show_tokens=_show_tokens)
        except TypeError:
            print("sentence must be list!")


class DataProcess(object):
    def __init__(self, sentence_list, pretrain_mode="",
                 max_length=max_sentence_len):
        self.sentence_list = sentence_list
        self.embedding_model = EmbeddingPreTrain()
        self.batch_size = 32
        self.max_length = max_length
        self.pretrain_mode = pretrain_mode

        self.data_x = None
        self.data_y = None
        self.sentence_data = None

    def get_feature(self, session=None):
        data_x = []
        data_y = []
        sentence_data = []

        _sentence_pair_list = []

        for sentence in tqdm(self.sentence_list):
            label = [0] * self.max_length
            char_list = [''] * self.max_length

            for index, _ in enumerate(sentence):
                tmp_char, tmp_char_label = _
                if index < self.max_length:
                    label[index] = label_map[label_dict[tmp_char_label]]
                    char_list[index] = tmp_char

            data_y.append(label)
            sentence_data.append("".join(char_list))

            _sentence_pair = "".join(char_list)
            _sentence_pair_list.append(_sentence_pair)

            if len(_sentence_pair_list) == 32:
                if self.pretrain_mode == "elmo":
                    _ = list(self.embedding_model.get_output(_sentence_pair_list, _show_tokens=False))
                    data_x.extend(_)
                else:
                    data_x.extend(list(self.embedding_model.get_output(_sentence_pair_list, _show_tokens=False)))
                _sentence_pair_list = []

        if len(_sentence_pair_list) > 0:
            data_x.extend(list(self.embedding_model.get_output(_sentence_pair_list, _show_tokens=False)))

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        sentence_data = np.array(sentence_data)

        self.data_x = data_x
        self.data_y = data_y
        self.sentence_data = sentence_data

        print("data_x shape:", data_x.shape)
        print("data_y shape:", data_y.shape)
        print("sentence_data shape:", sentence_data.shape)

    def next_batch(self):
        counter = 0
        batch_x = []
        batch_y = []
        batch_sen = []

        for (_x, _y, _sen) in zip(self.data_x, self.data_y, self.sentence_data):
            if counter == 0:
                batch_x = []
                batch_y = []
                batch_sen = []

            batch_x.append(_x)
            batch_y.append(_y)
            batch_sen.append(_sen)

            counter += 1

            if counter == self.batch_size:
                counter = 0
                yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)
        yield np.array(batch_sen), np.array(batch_x), np.array(batch_y)

    def get_one_sentence_feature(self, sentence):
        data_x = []
        data_y = []
        data_x.extend(list(self.embedding_model.get_output([sentence], _show_tokens=False)))
        data_y.append([0 for _ in range(self.max_length)])
        return np.array(data_x), np.array(data_y, dtype=np.int64)

    def get_sentence_list_feature(self, sentence_list):
        data_x = []
        data_y = []
        data_x.extend(list(self.embedding_model.get_output(sentence_list, _show_tokens=False)))
        [data_y.append([0 for _ in range(self.max_length)]) for i in sentence_list]
        return np.array(data_x), np.array(data_y, dtype=np.int64)


if __name__ == '__main__':
    from CRFSuiteForNER import *

    df = pd.read_csv('./bilmelmo/data/crfsuite_task1_train_bert.txt', quoting=csv.QUOTE_NONE,
                     encoding="utf-8", sep='\t', header=None)
    df.columns = ['Sentence #', 'word', 'tag']
    df = df.fillna(method='ffill')

    getter = SentenceGetter(df)
    sentences = getter.sentences

    a = DataProcess(sentence_list=sentences, max_length=128,
                    pretrain_mode="elmo")
    a.get_feature()

    for _, batch_x, batch_y in a.next_batch():
        print(batch_x.shape, batch_y.shape)
