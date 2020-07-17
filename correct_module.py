#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import json
import pandas as pd


def count_train_in_vocab(train_path, vocab_path):
    df = pd.read_csv(vocab_path, sep='\t', header=None)
    df.columns = ['word', 'type']
    a = df.groupby('type')['word'].apply(list)

    counter = 0
    in_counter = 0
    with open(train_path, mode='r', encoding="utf-8") as f1:
        for line in f1.readlines():
            sample = json.loads(line.strip())

            # 每个样本
            label_unit = dict()
            label_unit['text'] = sample['originalText']
            label_unit['intent'] = 'train_{}'.format(str(counter))
            counter += 1

            for entity in sample['entities']:
                # 每个实体
                text = sample['originalText'][entity['start_pos']:entity['end_pos']]
                label_type = entity['label_type']

                if text in a[label_type]:
                    in_counter += 1
                else:
                    print("{}@@{}".format(text, label_type))
                counter += 1
    print(counter, in_counter)


if __name__ == '__main__':
    count_train_in_vocab(train_path="./submit3.txt",
                         vocab_path="D:/data_file/ccks2020_2_task1_train/task1_vocab_new.txt")
