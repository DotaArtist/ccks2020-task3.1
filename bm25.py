#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import time
import pandas as pd
from gensim.summarization.bm25 import *


class MyBm25(BM25):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.corpus = corpus

    # def initialize(self):
    #     for document in self.corpus:
    #         frequencies = {}
    #         self.doc_len.append(len(document))
    #         for word in list(document):
    #             if word not in frequencies:
    #                 frequencies[word] = 0
    #             frequencies[word] += 1
    #         self.doc_freqs.append(frequencies)
    #
    #         for word, freq in iteritems(frequencies):
    #             if word not in self.df:
    #                 self.df[word] = 0
    #             self.df[word] += 1
    #
    #     for word, freq in iteritems(self.df):
    #         self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
    #

    def get_scores(self, document):
        scores = [self.get_score(document, index) for index in range(self.corpus_size)]
        scores_dict = dict(zip(range(self.corpus_size), scores))
        scores_out = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        # return self.corpus[scores_out[0][0]]
        _out = sorted([self.corpus[i[0]] for i in scores_out if i[1] > 0][:5],
                      key=lambda x: len(x), reverse=True)
        return sorted(list(set(_out)), key=lambda x: len(x), reverse=True)


def build_model():
    out_model = dict()
    _config = pd.read_csv("D:/data_file/ccks2020_2_task1_train/task1_vocab_total_add_train.txt", sep='\t', index_col=None, header=None)
    _config.columns = ['word', 'type']
    disease_corpus = _config[_config['type'] == '疾病和诊断'].iloc[:, 0].values.tolist()
    check_corpus = _config[_config['type'] == '影像检查'].iloc[:, 0].values.tolist()
    body_corpus = _config[_config['type'] == '解剖部位'].iloc[:, 0].values.tolist()
    operation_corpus = _config[_config['type'] == '手术'].iloc[:, 0].values.tolist()
    analysis_corpus = _config[_config['type'] == '实验室检验'].iloc[:, 0].values.tolist()
    drug_corpus = _config[_config['type'] == '药物'].iloc[:, 0].values.tolist()

    out_model['疾病和诊断'] = MyBm25(corpus=disease_corpus)
    out_model['影像检查'] = MyBm25(corpus=check_corpus)
    out_model['解剖部位'] = MyBm25(corpus=body_corpus)
    out_model['手术'] = MyBm25(corpus=operation_corpus)
    out_model['实验室检验'] = MyBm25(corpus=analysis_corpus)
    out_model['药物'] = MyBm25(corpus=drug_corpus)

    return out_model


if __name__ == '__main__':
    out_model = build_model()
    query_str = '头孢哌酮钠/舒巴坦钠'
    query = list(query_str)
    scores = out_model['药物'].get_scores(query)

    print(scores)
