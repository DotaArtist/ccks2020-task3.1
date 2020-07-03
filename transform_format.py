#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import json


# platform url:  https://rasahq.github.io/rasa-nlu-trainer/

ccks2019_data = "D:/data_file/ccks2019/ccks2019.json"
train_data = "D:/data_file/ccks2020_2_task1_train/task1_train.txt"
train_unlabel = "D:/data_file/ccks2020_2_task1_train/task1_unlabeled.txt"


def transform_ccks_platform(ccks_path, platform_path):
    """ccks 转 标注平台"""
    tools_data = dict()
    tools_data['rasa_nlu_data'] = dict()
    common_examples = []

    counter = 0
    with open(ccks_path, mode='r', encoding="utf-8") as f1:
        data = json.loads(f1.read())

        for sample in data:

            # 每个样本
            label_unit = dict()
            label_unit['text'] = sample['text']
            label_unit['intent'] = 'ccks2019_{}'.format(str(counter))
            counter += 1
            entities = []

            for entity in sample['mention']:

                # 每个实体
                entity_unit = dict()
                entity_unit['start'] = entity[1]
                entity_unit['end'] = entity[1] + len(entity[0])
                entity_unit['value'] = entity[0]
                entity_unit['entity'] = entity[2]
                entities.append(entity_unit)

            label_unit['entities'] = entities
            common_examples.append(label_unit)

        tools_data['rasa_nlu_data']['common_examples'] = common_examples

    with open(platform_path, 'w', encoding="utf-8") as fj:
        tools_data_str = json.dumps(tools_data, indent=4, ensure_ascii=False)
        fj.write(tools_data_str)


def transform_train_platform(train_path, platform_path):
    """训练数据 转 标注平台"""
    tools_data = dict()
    tools_data['rasa_nlu_data'] = dict()
    common_examples = []

    counter = 0
    with open(train_path, mode='r', encoding="utf-8") as f1:
        for line in f1.readlines():
            sample = json.loads(line.strip())

            # 每个样本
            label_unit = dict()
            label_unit['text'] = sample['originalText']
            label_unit['intent'] = 'train_{}'.format(str(counter))
            counter += 1
            entities = []

            for entity in sample['entities']:

                # 每个实体
                entity_unit = dict()
                entity_unit['start'] = entity['start_pos']
                entity_unit['end'] = entity['end_pos']
                entity_unit['value'] = sample['originalText'][entity['start_pos']:entity['end_pos']]
                entity_unit['entity'] = entity['label_type']
                entities.append(entity_unit)

            label_unit['entities'] = entities
            common_examples.append(label_unit)

        tools_data['rasa_nlu_data']['common_examples'] = common_examples

    with open(platform_path, 'w', encoding="utf-8") as fj:
        tools_data_str = json.dumps(tools_data, indent=4, ensure_ascii=False)
        fj.write(tools_data_str)


if __name__ == '__main__':
    # transform_ccks_platform(ccks_path=ccks2019_data, platform_path='ccks19_platform.json')
    transform_train_platform(train_path=train_data, platform_path='train_platform.json')
