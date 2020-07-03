#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import json
import numpy as np
import pandas as pd
from config import train_data


pd.set_option('display.max_rows', 100)

label_dict = {'疾病和诊断': 1, '影像检查': 3, '解剖部位': 5, '手术': 7, '药物': 9, '实验室检验': 11}


def transform_entities_to_label(text, entities):
    out = np.array([0 for i in range(len(text))])

    for i in entities:
        out[i["start_pos"]:i["end_pos"]] = label_dict[i["label_type"]]
        out[i["end_pos"]-1] = label_dict[i["label_type"]] + 1
    return out


length_list = []
with open(train_data, mode="r", encoding="utf-8") as f1:
    for line in f1.readlines():
        data = json.loads(line.strip())

        originalText = data["originalText"]
        entities = data["entities"]

        for i in entities:
            if i["label_type"] not in label_dict:
                label_dict[i["label_type"]] = len(label_dict)

print(label_dict)

