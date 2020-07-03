#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'


import json

ccks2019_data = "D:/data_file/ccks2017/ccks2017.json"
vocab_2019 = "D:/data_file/ccks2019/type_vocab.txt"

type_map = {
    'disease': "疾病和诊断",
    'diagnosis': "手术",
    'drug': "药物",
    'symptom': "症状",
}


def transform_format(_):
    _out = dict()
    _out["originalText"] = _["text"]
    _out["entities"] = []

    for _i in _["mention"]:
        pass


# 遍历生成vocab
with open(ccks2019_data, mode='r', encoding="utf-8") as f1:
    data = json.loads(f1.read())

    with open("./tmp_vocab.txt", mode="w", encoding="utf-8") as ft:
        for i in data:
            for j in i["mention"]:
                ft.writelines("{}\t{}\n".format(j[0], type_map[j[2]]))
