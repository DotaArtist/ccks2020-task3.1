#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1.ccks 格式 转 RASA；
2.训练数据 转 RASA；
3.crfsuite 转 RASA；

4.RASA 转 训练数据；
5.RASA 转 crf;
6.RASA 转 crfsuite;
"""

__author__ = 'yp'

import os
import json


# platform url:  https://rasahq.github.io/rasa-nlu-trainer/

ccks2019_data = "D:/data_file/ccks2019/ccks2019.json"
crf_data = "D:/data_file/ccks2020_2_task1_train/crf_train_demo.txt"

train_data = "D:/data_file/ccks2020_2_task1_train/task1_train.txt"
train_unlabel = "D:/data_file/ccks2020_2_task1_train/task1_unlabeled.txt"

platform_data = "./train_platform.json"


def transform_ccks_platform(ccks_path, platform_path):
    """1.ccks 格式 转 标注平台；"""
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
    """2.训练数据 转 标注平台；"""
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


def transform_crf_platform(crf_path, platform_path):
    """3.crfsuite 转 标注平台；"""

    def parse_label(_label_list, _char_list):
        """
        :param _label_list: [B-*, I-*, O-O...]
        :param _char_list:
        :return:
        """
        _entities = []
        entity_unit = dict()
        _text = "".join(_char_list)

        for _i, _j in enumerate(zip(_label_list, _char_list)):
            if _j[0].split('-')[0] == 'B':
                entity_unit['start'] = _i
                entity_unit['entity'] = _j[0].split('-')[1]

            if _i <= len(_char_list) - 2 and \
                    'start' in entity_unit.keys() and \
                    _label_list[_i + 1].split('-')[0] in ['B', 'O']:
                entity_unit['end'] = _i + 1
                entity_unit['value'] = _text[entity_unit['start']:entity_unit['end']]
                _entities.append(entity_unit)

                entity_unit = dict()

        if entity_unit and 'start' in entity_unit.keys():
            entity_unit['end'] = _i + 1
            entity_unit['value'] = _text[entity_unit['start']:entity_unit['end']]
            _entities.append(entity_unit)

        return _entities

    tools_data = dict()
    tools_data['rasa_nlu_data'] = dict()
    common_examples = []

    counter = 0
    with open(crf_path, mode='r', encoding="utf-8") as f1:
        label_unit = dict()  # 单个文档
        char_list = []
        label_list = []

        for line in f1.readlines():
            if line.strip() == "" and len(char_list) > 0:

                label_unit['intent'] = 'train_{}'.format(str(counter))
                label_unit['text'] = "".join(char_list)
                label_unit['entities'] = parse_label(label_list, char_list)

                common_examples.append(label_unit)

                counter += 1

                label_unit = dict()
                char_list = []
                label_list = []
            else:
                char, label = line.strip().split("\t")
                char_list.append(char)
                label_list.append(label)

        tools_data['rasa_nlu_data']['common_examples'] = common_examples

    with open(platform_path, 'w', encoding="utf-8") as fj:
        tools_data_str = json.dumps(tools_data, indent=4, ensure_ascii=False)
        fj.write(tools_data_str)


def transform_platform_train(train_path, platform_path):
    """4.RASA 转 训练数据；"""
    if not os.path.exists(train_path):
        with open(train_path, mode='w', encoding="utf-8") as fp:
            with open(platform_path, mode='r', encoding="utf-8") as f1:
                tools_data = json.loads(f1.read())

                for i in tools_data['rasa_nlu_data']['common_examples']:
                    label_unit = dict()
                    label_unit['originalText'] = i['text']
                    entities = []

                    for j in i["entities"]:
                        entity_unit = dict()
                        entity_unit['start_pos'] = j['start']
                        entity_unit['end_pos'] = j['end']
                        entity_unit['label_type'] = j['entity']
                        entities.append(entity_unit)
                    label_unit['entities'] = entities

                    fp.writelines("{}\n".format(json.dumps(label_unit, ensure_ascii=False)))
    else:
        print("file exists!!!")


def transform_platform_crf(crf_path, platform_path):
    """5.RASA 转 crfsuite;"""
    if not os.path.exists(crf_path):
        with open(crf_path, mode='w', encoding="utf-8") as fp:
            with open(platform_path, mode='r', encoding="utf-8") as f1:
                tools_data = json.loads(f1.read())

                for i in tools_data['rasa_nlu_data']['common_examples']:
                    text = i['text']
                    text_list = list(text)
                    text_label = ['O-O'] * len(text)

                    for j in i["entities"]:
                        text_label[j['start']: j['end']] = ['I-{}'.format(j['entity'])] * len(j['value'])
                        text_label[j['start']] = 'B-{}'.format(j['entity'])

                    _line_list = []
                    for _i, _j in zip(text_list, text_label):
                        _line_list.append("{}\t{}\n".format(_i, _j))

                    fp.writelines("{}\n".format("".join(_line_list)))
    else:
        print("file exists!!!")


def transform_platform_crfsuite(crf_path, platform_path):
    """6.RASA 转 crfsuite;"""
    if not os.path.exists(crf_path):
        with open(crf_path, mode='w', encoding="utf-8") as fp:
            with open(platform_path, mode='r', encoding="utf-8") as f1:
                tools_data = json.loads(f1.read())

                counter = 0
                for i in tools_data['rasa_nlu_data']['common_examples']:
                    text = i['text']
                    text_list = list(text)
                    text_label = ['O-O'] * len(text)

                    for j in i["entities"]:
                        text_label[j['start']: j['end']] = ['I-{}'.format(j['entity'])] * len(j['value'])
                        text_label[j['start']] = 'B-{}'.format(j['entity'])

                    _line_list = []
                    for _i, _j in zip(text_list, text_label):
                        if len(_line_list) == 0:
                            _line_list.append("{}\t{}\n".format(_i, _j))
                        else:
                            _line_list.append("\t{}\t{}\n".format(_i, _j))

                    fp.writelines("Sentence_{}\t{}".format(str(counter), "".join(_line_list)))
                    counter += 1
    else:
        print("file exists!!!")


if __name__ == '__main__':
    # transform_ccks_platform(ccks_path=ccks2019_data, platform_path='ccks19_platform.json')
    # transform_train_platform(train_path=train_data, platform_path='train_platform.json')
    # transform_crf_platform(crf_path='tmp.txt', platform_path='tmp1.json')

    # transform_platform_train(platform_path=platform_data, train_path='tmp.txt')
    # transform_platform_crf(platform_path='train_platform.json', crf_path='crf_train.txt')
    transform_platform_crfsuite(platform_path='train_platform.json', crf_path='crf_train.txt')
