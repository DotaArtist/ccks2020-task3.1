#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'


import json

ccks2019_data = "D:/data_file/ccks2020_2_task1_train/ccks2_task1_val/task1_no_val.txt"
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
with open(ccks2019_data, mode='r', encoding="gbk") as f1:
    for line in f1.readlines():
        data = json.loads(line)
        print(data["originalText"])

    # with open("./tmp_vocab.txt", mode="w", encoding="utf-8") as ft:
    #     for i in data:
    #         for j in i["mention"]:
                # ft.writelines("{}\t{}\n".format(j[0], type_map[j[2]]))
