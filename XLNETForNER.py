#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'

import os
import gc
import json
import time
import torch
import random
import numpy as np
from apex import amp
import datetime
from transformers import AdamW
from transformers import AutoConfig
from transformers import XLNetTokenizer
from transformers import XLNetForTokenClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from seqeval.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

max_token_length = 1000
batch_size = 1
epochs = 1

train_data = "D:/data_file/ccks2020_2_task1_train/task1_train.txt"
# train_data = "D:/data_file/ccks2020_2_task1_train/task1_train_demo.txt"
model_path = "D:/model_file/hfl_chinese-xlnet-base"
xlnet_out_address = "D:/model_file/my_xlnet"
predict_mode = False
predict_data = "D:/data_file/ccks2020_2_task1_train/task1_train_demo.txt"

label_dict = {'疾病和诊断': 1, '影像检查': 3, '解剖部位': 5, '手术': 7, '药物': 9, '实验室检验': 11}
label_map_reverse = {
    0: 'O',
    1: 'B-disease',
    2: 'I-disease',
    3: 'B-check',
    4: 'I-check',
    5: 'B-part',
    6: 'I-part',
    7: 'B-operation',
    8: 'I-operation',
    9: 'B-drug',
    10: 'I-drug',
    11: 'B-assay',
    12: 'I-assay',
}

unk_token = '<unk>'


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(_preds, _labels):
    pred_flat = np.argmax(_preds, axis=2).flatten()
    labels_flat = _labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_label_name(_labels):
    """
    :param _labels: [[],[],[]]
    :return:
    """
    return [[label_map_reverse[i] for i in j] for j in _labels]


def process_text(text):
    text = text.replace("Ⅰ", '一')
    text = text.replace("Ⅱ", '二')
    text = text.replace("Ⅲ", '三')
    text = text.replace("Ⅳ", '四')
    text = text.replace("Ⅴ", '五')
    text = text.replace("Ⅵ", '六')
    text = text.replace("Ⅶ", '七')
    text = text.replace("Ⅷ", '八')
    text = text.replace("Ⅸ", '九')
    text = text.replace("Ⅹ", '十')
    text = text.replace("℃", "度")
    text = text.replace("㎝", "m")
    text = text.replace("㎡", "m")

    return text


def transform_entities_to_label(text, entities, sep_sentence):
    """
    原文，实体，wwm ===> 标签
    :param text: originalText
    :param entities: [{"start_pos": 10, "end_pos": 13, "label_type": "疾病和诊断"}]
    :param sep_sentence: [word, word]
    :return: [0, 0, 1, 0]
    """
    char_label = np.array([0 for i in range(len(text))])
    out = np.array([0 for i in range(max_token_length)])

    for i in entities:
        char_label[i["start_pos"]:i["end_pos"]] = label_dict[i["label_type"]]
        char_label[i["end_pos"] - 1] = label_dict[i["label_type"]] + 1

    current_idx = 0
    pad_num = 0
    while '<pad>' in sep_sentence:
        sep_sentence.remove("<pad>")
        pad_num += 1

    for i, j in enumerate(sep_sentence[1:-2]):
        out[i + pad_num + 1] = max(char_label[current_idx:current_idx + len(j)])

        if j == "<unk>":
            current_idx = current_idx + 1
        else:
            current_idx = current_idx + len(j)

    return out.tolist()


config = AutoConfig.from_pretrained(model_path)
tokenizer = XLNetTokenizer.from_pretrained(model_path, unk_token=unk_token)
model = XLNetForTokenClassification.from_pretrained(model_path, num_labels=13)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
# device = torch.device("cpu")

model.to(device)

train_input_ids = []
train_labels = []
train_masks = []

with open(train_data, mode="r", encoding="utf-8") as f1:
    for line in f1.readlines():
        data = json.loads(line.strip())

        originalText = process_text(data["originalText"])
        entities = data["entities"]

        input_ids = torch.tensor(
            tokenizer.encode(originalText,
                             add_special_tokens=True,
                             pad_to_max_length=True,
                             max_length=max_token_length)).unsqueeze(0)
        sep_sentence = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
        seq_lab = transform_entities_to_label(originalText, entities, sep_sentence)
        labels = torch.tensor(seq_lab).unsqueeze(0)

        encoded_dict = tokenizer.encode_plus(
            originalText,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_token_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        train_masks.append(encoded_dict["attention_mask"])
        train_input_ids.append(input_ids)
        train_labels.append(labels)

        print(len([_ for _ in input_ids.reshape(-1).tolist() if _ != 5]))
assert 1==2
train_input_ids = torch.cat(train_input_ids, dim=0)
train_labels = torch.cat(train_labels, dim=0)
train_masks = torch.cat(train_masks, dim=0)

train_dataset = TensorDataset(train_input_ids, train_labels, train_masks)
# train_set, val_set = torch.utils.data.random_split(train_dataset, [950, 100])
train_set, val_set = torch.utils.data.random_split(train_dataset, [15, 5])

train_data_loader = DataLoader(
    train_set,
    sampler=RandomSampler(train_set),
    batch_size=batch_size
)

val_data_loader = DataLoader(
    val_set,
    sampler=SequentialSampler(val_set),
    batch_size=batch_size
)

params = list(model.named_parameters())

print('The model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

optimizer = AdamW(model.parameters())

total_steps = len(train_data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

# FP16
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_data_loader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data_loader), elapsed))

        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        b_input_mask = batch[2].to(device)

        model.zero_grad()

        loss, logits = model(b_input_ids, labels=b_labels, attention_mask=b_input_mask)

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        gc.collect()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_data_loader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_loss = 0
    nb_eval_steps = 0

    val_pred = []
    val_lab = []

    # Evaluate data for one epoch
    for batch in val_data_loader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        b_masks = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids, labels=b_labels, attention_mask=b_masks)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        val_pred.append(np.argmax(logits, axis=2).flatten().tolist())
        val_lab.append(label_ids.flatten().tolist())

    # Report the final accuracy for this validation run.
    val_pred_tag = get_label_name(val_pred)
    val_lab_tag = get_label_name(val_lab)

    avg_val_accuracy = accuracy_score(y_pred=val_pred_tag, y_true=val_lab_tag)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    val_class_report = classification_report(y_pred=val_pred_tag, y_true=val_lab_tag)
    print("====class_report:", val_class_report)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_data_loader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

print("save model!")
if not os.path.exists(xlnet_out_address):
        os.makedirs(xlnet_out_address)

model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(xlnet_out_address, "pytorch_model.bin")
output_config_file = os.path.join(xlnet_out_address, "config.json")

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(xlnet_out_address)


if predict_mode:
    model = XLNetForTokenClassification.from_pretrained(xlnet_out_address, num_labels=13)
    tokenizer = XLNetTokenizer.from_pretrained(xlnet_out_address)
    gc.collect()
    model.to(device)

    predict_input_ids = []
    predict_masks = []

    with open(predict_data, mode="r", encoding="utf-8") as f1:
        for line in f1.readlines():
            data = line.strip()

            originalText = process_text(data)

            input_ids = torch.tensor(
                tokenizer.encode(originalText,
                                 add_special_tokens=True,
                                 pad_to_max_length=True,
                                 max_length=max_token_length)).unsqueeze(0)
            sep_sentence = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
            labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)
            outputs = model(input_ids.to(device), labels=labels.to(device))
            print(np.argmax(outputs[1].detach().cpu().numpy(), axis=2).flatten())
            gc.collect()
