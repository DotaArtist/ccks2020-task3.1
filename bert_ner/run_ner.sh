#!/usr/bin/env bash

  python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --crf=True \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=mydata   \
    --vocab_file=D:/model_file/chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=D:/model_file/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=D:/model_file/chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=256   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=4.0   \
    --output_dir=./output/result_dir


perl conlleval.pl -d '\t' < ./output/result_dir/label_test.txt

# bert fine-tuning 短句128
python BERT_NER.py --task_name="NER" --do_lower_case=True --crf=True --do_train=False --do_eval=False --do_predict=True --data_dir=mydata --vocab_file=D:/model_file/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=D:/model_file/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=D:/model_file/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=2e-5 --num_train_epochs=4.0 --output_dir=./output/result_dir
