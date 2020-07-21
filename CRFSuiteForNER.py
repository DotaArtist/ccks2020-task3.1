#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import csv
import json
import pickle
import string
import pandas as pd
import scipy.stats
from time import time
import flashtext
import sklearn_crfsuite
from collections import Counter
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


# https://github.com/susanli2016/NLP-with-Python/blob/master/NER_sklearn.ipynb

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

# label_dict = {
#     'B-疾病和诊断': 'B-disease',
#     'B-影像检查': 'O',
#     'B-解剖部位': 'O',
#     'B-手术': 'O',
#     'B-药物': 'O',
#     'B-实验室检验': 'O',
#     'I-疾病和诊断': 'I-disease',
#     'I-影像检查': 'O',
#     'I-解剖部位': 'O',
#     'I-手术': 'O',
#     'I-药物': 'O',
#     'I-实验室检验': 'O',
#     'O-O': 'O',
# }


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

label_dict_reverse = {v: k for k, v in label_dict.items()}

lll = list(label_dict.values())
lll.remove('O')

punc = string.punctuation + '：，。？、“”'


class Node(object):

    def __init__(self):
        self.next = {}  # 相当于指针，指向树节点的下一层节点
        self.fail = None  # 失配指针，这个是AC自动机的关键
        self.isWord = False  # 标记，用来判断是否是一个标签的结尾
        self.word = ""


class AcAutomation(object):

    def __init__(self):
        self.root = Node()

    def add_keyword(self, word):
        temp_root = self.root
        for char in word:
            if char not in temp_root.next:
                temp_root.next[char] = Node()
            temp_root = temp_root.next[char]
        temp_root.isWord = True
        temp_root.word = word

    def make_fail(self):
        temp_que = []
        temp_que.append(self.root)
        while len(temp_que) != 0:
            temp = temp_que.pop(0)
            p = None
            for key, value in temp.next.item():
                if temp == self.root:
                    temp.next[key].fail = self.root
                else:
                    p = temp.fail
                    while p is not None:
                        if key in p.next:
                            temp.next[key].fail = p.fail
                            break
                        p = p.fail
                    if p is None:
                        temp.next[key].fail = self.root
                temp_que.append(temp.next[key])

    def extract_keywords(self, content):
        p = self.root
        result = []
        current_position = 0

        while current_position < len(content):
            word = content[current_position]
            while word in p.next == False and p != self.root:
                p = p.fail

            if word in p.next:
                p = p.next[word]
            else:
                p = self.root

            if p.isWord:
                result.append(p.word)
            else:
                current_position += 1
        return result


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s['word'].values.tolist(),
                                                     s['tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence_{}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def get_vocab(_vocab, _type):
    """
    :param _vocab:
    :param _type: 疾病和诊断,手术,药物,解剖部位,实验室检验,影像检查
    :return:
    """
    tmp_vocab = _vocab[_vocab['类型'] == _type]['名称'].tolist()
    _ = sorted(Counter(list("".join(tmp_vocab))).items(), key=lambda x: x[1], reverse=False)
    return [i[0] for i in _]


df = pd.read_csv('crf_train.txt', quoting=csv.QUOTE_NONE,
                 encoding="utf-8", sep='\t', header=None)
df.columns = ['Sentence #', 'word', 'tag']
df = df.fillna(method='ffill')
getter = SentenceGetter(df)
sentences = getter.sentences

# df = pd.read_csv('crf_ner_2w_checked.txt', quoting=csv.QUOTE_NONE,
#                  encoding="utf-8", sep='\t', header=None)
# df.columns = ['Sentence #', 'word', 'tag']
# df = df.fillna(method='ffill')
# getter = SentenceGetter(df)
# sentences = getter.sentences
#
# df = pd.read_csv('crf_train_v3.txt', quoting=csv.QUOTE_NONE,
#                  encoding="utf-8", sep='\t', header=None)
# df.columns = ['Sentence #', 'word', 'tag']
# df = df.fillna(method='ffill')
# getter = SentenceGetter(df)
# sentences.extend(getter.sentences)


def word2features(sent, i):
    """
    处理每句中每个字
    """
    word = sent[i][0]
    features = [
        'bias',
        'word=' + word,
        #'word.ispunc=%s' % (word in punc),
        #'word.isdigit=%s' % word.isdigit(),
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        features.extend([
            '-1:word=' + word1,
         #   '-1:word.ispunc=%s' % (word1 in punc),
         #   '-1:word.isdigit=%s' % word1.isdigit(),
        ])
        if i > 1:
            word1 = sent[i - 2][0]
            features.extend([
                '-2:word=' + word1,
          #      '-2:word.ispunc=%s' % (word1 in punc),
           #     '-2:word.isdigit=%s' % word1.isdigit(),
            ])

    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.extend([
            '+1:word=' + word1,
            #'+1:word.ispunc=%s' % (word1 in punc),
            #'+1:word.isdigit=%s' % word1.isdigit(),
        ])
        if i < len(sent) - 2:
            word1 = sent[i + 2][0]
            features.extend([
                '+2:word=' + word1,
             #   '+2:word.ispunc=%s' % (word1 in punc),
              #  '+2:word.isdigit=%s' % word1.isdigit(),
            ])
    else:
        features.append('EOS')
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label_dict[label] for token, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=123)


def train_and_save():
    print('Training...')
    start_time = time()
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                               # c1=0.3795835381454335,
                               # c2=0.08194774957699179,
                               c1=0.1,
                               c2=0.1,
                               max_iterations=100,
                               all_possible_states=True,
                               all_possible_transitions=True)
    crf.fit(X_train, y_train)
    end_time = time()
    print('Train is over! It takes {} s.'.format('%.2f' % (end_time - start_time)))

    # 保存模型:
    with open('mycrf.pickle', 'wb') as f:
        pickle.dump(crf, f)

    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred,
                                average='weighted',
                                labels=lll))

    print(metrics.flat_classification_report(y_test,
                                             y_pred,
                                             labels=lll))
    return crf


def tune_parameters():
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
        'all_possible_states': [True, False]
    }
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted',
                            labels=lll)
    rs = RandomizedSearchCV(crf,
                            params_space,
                            cv=10,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=100,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)


def predict(crf, sentence):
    _x = sent2features(sentence)
    pred_tag = crf.predict_single(_x)
    return pred_tag


def load_vocab_model(vocab_path):
    """加载词库"""
    _df = pd.read_csv(vocab_path, sep='\t', header=None)
    _df.columns = ['word', 'type']
    a = _df.groupby('type')['word'].apply(list)

    vocab_model = dict()
    for i in a.keys():
        _extractor = flashtext.KeywordProcessor()
        # _extractor = AcAutomation()
        for _key in a[i]:
            if len(_key) > 0:
                _extractor.add_keyword(_key.strip(" "))
        vocab_model[i] = _extractor
    return a, vocab_model


def vocab_predict(vocab_model, sentence):
    out_dict = dict()

    for i in vocab_model.keys():
        _model = vocab_model[i]
        out_dict[i] = _model.extract_keywords(sentence)
    return out_dict


if __name__ == '__main__':
    # tune_parameters()
    crf = train_and_save()

    with open('task1_unlabeled_predict.txt', mode='w', encoding="utf-8") as fp:
        with open('D:/data_file/ccks2020_2_task1_train/ccks2_task1_val/task1_no_val.txt', mode='r', encoding="gbk") as ft:
            for line in ft.readlines():
                line = json.loads(line.strip("\n"))
                pred_tag = predict(crf, line['originalText'])

                tmp = []
                for _i, _j in zip(list(line['originalText']), pred_tag):
                    tmp.append("{}\t{}\n".format(_i, label_dict_reverse[_j]))

                fp.writelines("{}\n".format("".join(tmp)))
