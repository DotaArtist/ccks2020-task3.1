#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1.ccks 格式 转 RASA；
2.训练数据 转 RASA；
3.crfsuite 转 RASA；

4.RASA 转 训练数据；
5.RASA 转 crf;
5.1.RASA 转 crfpp;
6.RASA 转 crfsuite;
6.1 RASA 转 crfsuite, 加词性;

7.raw 转 crfsuite;
8.nuanwa 转 RASA;
9. 训练结果过滤
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
                try:
                    char, label = line.strip("\n").split("\t")
                    char_list.append(char)
                    label_list.append(label)
                except ValueError:
                    print(line)
                    assert 1 == 2

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
    """5.RASA 转 crf;"""
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


def transform_platform_crfpp(crf_path, platform_path):
    """5.1.RASA 转 crfpp;"""
    if not os.path.exists(crf_path):
        with open(crf_path, mode='w', encoding="utf-8") as fp:
            with open(platform_path, mode='r', encoding="utf-8") as f1:
                tools_data = json.loads(f1.read())

                for i in tools_data['rasa_nlu_data']['common_examples']:
                    text = i['text']
                    text_list = list(text)
                    text_label = ['O'] * len(text)

                    for j in i["entities"]:
                        text_label[j['start']: j['end']] = ['{}+i'.format(j['entity'])] * len(j['value'])
                        text_label[j['start']] = '{}+b'.format(j['entity'])

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


import jieba.posseg as psg


def transform_platform_crfsuite_pos(crf_path, platform_path):
    """6.1 RASA 转 crfsuite, 加词性;"""
    if not os.path.exists(crf_path):
        with open(crf_path, mode='w', encoding="utf-8") as fp:
            with open(platform_path, mode='r', encoding="utf-8") as f1:
                tools_data = json.loads(f1.read())

                counter = 0
                for i in tools_data['rasa_nlu_data']['common_examples']:
                    text = i['text']
                    text_list = list(text)
                    text_label = ['O-O'] * len(text)

                    seg = psg.cut(text)
                    pos_list = []
                    for ele in seg:
                        _, __ = ele.word, ele.flag
                        for j in _:
                            pos_list.append(__)

                    for j in i["entities"]:
                        text_label[j['start']: j['end']] = ['I-{}'.format(j['entity'])] * len(j['value'])
                        text_label[j['start']] = 'B-{}'.format(j['entity'])

                    _line_list = []
                    for _i, _j, _k in zip(text_list, text_label, pos_list):
                        if len(_line_list) == 0:
                            _line_list.append("{}\t{}\t{}\n".format(_i, _k, _j))
                        else:
                            _line_list.append("\t{}\t{}\t{}\n".format(_i, _k, _j))

                    fp.writelines("Sentence_{}\t{}".format(str(counter), "".join(_line_list)))
                    counter += 1
    else:
        print("file exists!!!")


def transform_validation_crfsuite(crf_path, validation_path):
    """7.val 转 crfsuite;"""
    counter = 0
    if not os.path.exists(crf_path):
        with open(crf_path, mode='w', encoding="utf-8") as fp:
            with open(validation_path, mode='r', encoding="utf-8") as f1:
                for line in f1.readlines():
                    sample = json.loads(line.strip())

                    # 每个样本
                    label_unit = dict()
                    label_unit['text'] = sample['originalText']
                    # todo


def transform_nuanwa_platform(nuanwa_path, platform_path):
    """8.nuanwa 转 RASA;"""
    _map = {
        "disease": "疾病和诊断",
        "diagnosis": "手术",
        "drug": "药物",
    }

    def parse_label(text, ner_label):
        """
        :param _label_list: [B-*, I-*, O-O...]
        :param _char_list:
        :return:
        """
        _entities = []
        ner_label = ner_label.split(":")[1]

        if ner_label == "null":
            return _entities
        else:
            for i in ner_label.split("&&"):
                tmp_label, tmp_entity = i.split("@@")
                if tmp_entity in text:
                    entity_unit = dict()
                    entity_unit["start"] = text.index(tmp_entity)
                    entity_unit["end"] = text.index(tmp_entity) + len(tmp_entity)
                    entity_unit["value"] = tmp_entity

                    if tmp_label in _map.keys():
                        entity_unit["entity"] = _map[tmp_label]
                        _entities.append(entity_unit)
                else:
                    pass

        return _entities

    tools_data = dict()
    tools_data['rasa_nlu_data'] = dict()
    common_examples = []

    counter = 0
    with open(nuanwa_path, mode='r', encoding="utf-8") as f1:
        label_unit = dict()  # 单个文档

        for line in f1.readlines():
            text, ner_label = line.strip().split("\t")

            label_unit['intent'] = 'train_{}'.format(str(counter))
            label_unit['text'] = text
            label_unit['entities'] = parse_label(text, ner_label)

            common_examples.append(label_unit)

            counter += 1

            label_unit = dict()

        tools_data['rasa_nlu_data']['common_examples'] = common_examples

    with open(platform_path, 'w', encoding="utf-8") as fj:
        tools_data_str = json.dumps(tools_data, indent=4, ensure_ascii=False)
        fj.write(tools_data_str)


def transform_train_filter(train_path, train_filter_path):
    """9. 训练结果过滤"""
    from CRFSuiteForNER import load_vocab_model
    from CRFSuiteForNER import vocab_predict

    vocab_path = "D:/data_file/ccks2020_2_task1_train/task1_vocab_new.txt"
    vocab_model_dict, vocab_model = load_vocab_model(vocab_path)

    vocab_model_dict_reverse = dict()

    for i in vocab_model_dict.keys():
        for j in vocab_model_dict[i]:
            if j not in vocab_model_dict_reverse.keys():
                vocab_model_dict_reverse[j] = []

            if i not in vocab_model_dict_reverse[j]:
                vocab_model_dict_reverse[j].append(i)

    with open(train_filter_path, 'w', encoding="utf-8") as fj:
        with open(train_path, mode='r', encoding="utf-8") as f1:
            for line in f1.readlines():
                sample = json.loads(line.strip())
                new_sample = sample.copy()
                new_entities = []

                text = sample["originalText"]

                # 词库结果
                out_dict = vocab_predict(vocab_model, sentence=text)
                out_dict_reverse = dict()

                for i in out_dict.keys():
                    for j in out_dict[i]:
                        if j not in out_dict_reverse.keys():
                            out_dict_reverse[j] = []

                        if i not in out_dict_reverse[j]:
                            out_dict_reverse[j].append(i)

                added_start_list = [0] * len(text)

                sample_entities_list = sample["entities"]
                sample_entities_list.sort(key=lambda x: x['end_pos'] - x['start_pos'], reverse=True)
                for entity in sample_entities_list:
                    # 每个实体

                    filter_list = out_dict[entity["label_type"]]
                    origin_entity = text[entity["start_pos"]: entity["end_pos"]]

                    vocab_type_list = vocab_model_dict_reverse.get(origin_entity, [])

                    # 1. 词库/模型双重匹配
                    if origin_entity in filter_list \
                            and sum(added_start_list[entity["start_pos"]:entity["end_pos"]]) < len(origin_entity):
                        new_entities.append(entity)
                        added_start_list[entity["start_pos"]: entity["end_pos"]] = [1] * len(origin_entity)
                        print("0@@{}@@{}".format(origin_entity, entity["label_type"]))

                    # 2.
                    elif entity["label_type"] in ["药物"] \
                            and sum(added_start_list[entity["start_pos"]:entity["end_pos"]]) < len(origin_entity):
                        new_entities.append(entity)
                        added_start_list[entity["start_pos"]: entity["end_pos"]] = [1] * len(origin_entity)
                        print("2@@{}@@{}".format(origin_entity, entity["label_type"]))

                    # 3. 词库补充，其他类别（修正错误类别）
                    elif len(vocab_type_list) == 1 \
                            and sum(added_start_list[entity["start_pos"]:entity["end_pos"]]) < len(origin_entity):
                        entity["label_type"] = vocab_type_list[0]
                        print("3@@{}@@{}".format(origin_entity, entity["label_type"]))
                        new_entities.append(entity)
                        added_start_list[entity["start_pos"]: entity["end_pos"]] = [1] * len(origin_entity)

                    # 4. 较长的实体召回
                    elif len(origin_entity) > 1 \
                            and entity["label_type"] in ["手术", "疾病和诊断", "影像检查", "实验室检验"]:
                        # new_entities.append(entity)
                        # added_start_list[entity["start_pos"]: entity["end_pos"]] = [1] * len(origin_entity)
                        print("4@@{}@@{}".format(origin_entity, entity["label_type"]))
                    else:
                        print("-1@@{}@@{}".format(origin_entity, entity["label_type"]))

                # 5. 词库完全召回
                _entity_list = list(out_dict_reverse.keys())
                _entity_list = sorted(_entity_list, key=len, reverse=True)
                new_text = text

                for _entity in _entity_list:
                    _entity = _entity.strip(" ")
                    if _entity in new_text \
                            and len(out_dict_reverse[_entity]) == 1 \
                            and out_dict_reverse[_entity][0] in ["药物"]:
                        tmp_start = new_text.index(_entity)
                        if sum(added_start_list[tmp_start:tmp_start + len(_entity)]) < len(_entity):
                            entity_tmp = dict()
                            entity_tmp["label_type"] = out_dict_reverse[_entity][0]
                            entity_tmp["start_pos"] = new_text.index(_entity)
                            entity_tmp["end_pos"] = entity_tmp["start_pos"] + len(_entity)
                            new_entities.append(entity_tmp)
                            added_start_list[entity_tmp["start_pos"]: entity_tmp["end_pos"]] = [1] * len(_entity)
                            print("5@@{}@@{}".format(_entity, entity_tmp["label_type"]))

                            text_tmp_list = list(new_text)
                            text_tmp_list[entity_tmp["start_pos"]:entity_tmp["end_pos"]] = ['@'] * len(_entity)
                            new_text = "".join(text_tmp_list)
                        else:
                            pass

                    elif _entity in new_text \
                            and len(out_dict_reverse[_entity]) == 1 \
                            and len(_entity) > 3 \
                            and out_dict_reverse[_entity][0] in ["手术"]:
                        tmp_start = new_text.index(_entity)
                        if sum(added_start_list[tmp_start:tmp_start + len(_entity)]) < len(_entity):
                            entity_tmp = dict()
                            entity_tmp["label_type"] = out_dict_reverse[_entity][0]
                            entity_tmp["start_pos"] = new_text.index(_entity)
                            entity_tmp["end_pos"] = entity_tmp["start_pos"] + len(_entity)
                            new_entities.append(entity_tmp)
                            added_start_list[entity_tmp["start_pos"]: entity_tmp["end_pos"]] = [1] * len(_entity)
                            print("5@@{}@@{}".format(_entity, entity_tmp["label_type"]))

                            text_tmp_list = list(new_text)
                            text_tmp_list[entity_tmp["start_pos"]:entity_tmp["end_pos"]] = ['@'] * len(_entity)
                            new_text = "".join(text_tmp_list)
                        else:
                            pass

                new_sample["entities"] = new_entities
                fj.writelines("{}\n".format(json.dumps(new_sample, ensure_ascii=False)))


if __name__ == '__main__':
    # ccks 转 rasa
    # transform_ccks_platform(ccks_path=ccks2019_data, platform_path='ccks19_platform.json')

    # train 转 rasa
    # transform_train_platform(train_path='./submit12.txt', platform_path='submit12.json')

    # crf 转 rasa
    transform_crf_platform(crf_path='task1_unlabeled_predict.txt', platform_path='submit13.json')

    # rasa 转 train
    transform_platform_train(platform_path='submit13.json', train_path='submit13.txt')

    # rasa 转 crf
    # transform_platform_crf(platform_path='D:/data_file/ccks2020_2_task1_train/task1_train.json',
    #                        crf_path='crfpp_task1_train.txt')
    # transform_platform_crfpp(platform_path='D:/data_file/ccks2020_2_task1_train/task1_train.json',
    #                        crf_path='crfpp_task1_train.txt')

    # rasa 转 crfsuite
    # transform_platform_crfsuite(platform_path='./nuanwa_train.json', crf_path='crf_nuanwa_train.txt')

    # raw 转 crfsuite
    # transform_validation_crfsuite(crf_path="./test.txt", raw_path="D:/data_file/ccks2020_2_task1_train/ccks2_task1_val/task1_no_val.txt")

    # rasa 转
    # transform_platform_crfsuite_pos(crf_path='crf_pos_train.txt',
    #                                 platform_path="D:/data_file/ccks2020_2_task1_train/task1_train.json")

    # nuanwa 转 rasa
    # transform_nuanwa_platform(nuanwa_path="D:/data_file/ccks2020_2_task1_train/nuanwa_train.txt", platform_path="./nuanwa_train.json")

    # 训练数据 过滤
    # transform_train_filter(train_path='./提交/submit12.txt', train_filter_path='submit13.txt')
