#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import json


# https://rasahq.github.io/rasa-nlu-trainer/

ccks2019_data = "D:/data_file/ccks2019/ccks2019.json"
train_data = "D:/data_file/ccks2020_2_task1_train/task1_train.txt"
train_unlabel = "D:/data_file/ccks2020_2_task1_train/task1_unlabeled.txt"


# ccks -> 标注
def transform_ccks_platform():
    tools_data = dict()
    tools_data['rasa_nlu_data'] = dict()
    tools_data['rasa_nlu_data']['common_examples'] = []

    counter = 0
    with open(ccks2019_data, mode='r', encoding="utf-8") as f1:
        data = json.loads(f1.read())

        for sample in data:

            # 每个样本
            label_unit = dict()
            label_unit['text'] = sample['text']
            label_unit['intent'] = 'ccks_{}'.format(str(counter))
            label_unit['entities'] = []

            for entity in sample['mention']:

                # 每个实体
                entity_unit = dict()
                entity_unit['start'] = entity[1]
                entity_unit['end'] = entity[1] + len(entity[0])
                entity_unit['value'] = entity[0]
                entity_unit['entity'] = entity[2]
    json.dump(tools_data, 'platform.json')


if __name__ == '__main__':
    transform_ccks_platform()
