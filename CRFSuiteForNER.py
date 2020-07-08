#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import scipy.stats
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
from time import time
import pickle
import string

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

label_dict_reverse = {v: k for k, v in label_dict.items()}

lll = list(label_dict.values())
lll.remove('O')

punc = string.punctuation + '：，。？、“”'


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


df = pd.read_csv('crf_train.txt',
                 encoding="utf-8", sep='\t', header=None)
df.columns = ['Sentence #', 'word', 'tag']
df = df.fillna(method='ffill')

getter = SentenceGetter(df)
sentences = getter.sentences

vocab = pd.read_excel('vocab.xlsx',
                      sheet_name='vocab',
                      index_col=None,
                      header=0)


drug_vocab = get_vocab(_vocab=vocab, _type='药物')
body_vocab = get_vocab(_vocab=vocab, _type='解剖部位')
analysis_vocab = get_vocab(_vocab=vocab, _type='实验室检验')
check_vocab = get_vocab(_vocab=vocab, _type='影像检查')
operation_vocab = get_vocab(_vocab=vocab, _type='手术')
disease_vocab = get_vocab(_vocab=vocab, _type='疾病和诊断')


def word2features_old(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
    }
    features.update({
        '0:word.is_disease': 1. if word in disease_vocab else 0,
        '0:word.is_drug': 1. if word in drug_vocab else 0,
        '0:word.is_body': 1. if word in body_vocab else 0,
        '0:word.is_analysis': 1. if word in analysis_vocab else 0,
        '0:word.is_check': 1. if word in check_vocab else 0,
        '0:word.is_operation': 1. if word in operation_vocab else 0,
    })
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.is_disease': 1. if word1 in disease_vocab else 0,
            '-1:word.is_drug': 1. if word1 in drug_vocab else 0,
            '-1:word.is_body': 1. if word1 in body_vocab else 0,
            '-1:word.is_analysis': 1. if word1 in analysis_vocab else 0,
            '-1:word.is_check': 1. if word1 in check_vocab else 0,
            '-1:word.is_operation': 1. if word1 in operation_vocab else 0,
        })
    else:
        features['BOS'] = 1.
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.is_disease': 1. if word1 in disease_vocab else 0,
            '+1:word.is_drug': 1. if word1 in drug_vocab else 0,
            '+1:word.is_body': 1. if word1 in body_vocab else 0,
            '+1:word.is_analysis': 1. if word1 in analysis_vocab else 0,
            '+1:word.is_check': 1. if word1 in check_vocab else 0,
            '+1:word.is_operation': 1. if word1 in operation_vocab else 0,
        })
    else:
        features['EOS'] = 1.

    return features


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def train_and_save():
    print('Training...')
    start_time = time()
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
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


crf = train_and_save()

with open('task1_unlabeled_predict.txt', mode='w', encoding="utf-8") as fp:
    with open('D:/data_file/ccks2020_2_task1_train/task1_unlabeled.txt', mode='r', encoding="utf-8") as ft:
        for line in ft.readlines():
            pred_tag = predict(crf, line.strip("\n"))

            tmp = []
            for _i, _j in zip(list(line), pred_tag):
                tmp.append("{}\t{}\n".format(_i, label_dict_reverse[_j]))

            fp.writelines("{}\n".format("".join(tmp)))
