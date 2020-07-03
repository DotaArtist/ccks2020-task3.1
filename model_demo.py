#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""<>"""

__author__ = 'yp'


from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.configuration_albert import AlbertConfig
from transformers.tokenization_bert import BertTokenizer
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torch
from torch.optim import Optimizer
from transformers import AlbertForSequenceClassification
from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_bert import ACT2FN
from transformers.tokenization_bert import BertTokenizer
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class AlbertSequenceOrderHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 2)
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        prediction_scores = hidden_states + self.bias

        return prediction_scores


class AlbertForPretrain(AlbertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)

        # For Masked LM
        # The original huggingface implementation, created new output weights via dense layer
        # However the original Albert
        self.predictions_dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.predictions_activation = ACT2FN[config.hidden_act]
        self.predictions_LayerNorm = nn.LayerNorm(config.embedding_size)
        self.predictions_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.predictions_decoder = nn.Linear(config.embedding_size, config.vocab_size)

        self.predictions_decoder.weight = self.albert.embeddings.word_embeddings.weight

        # For sequence order prediction
        self.seq_relationship = AlbertSequenceOrderHead(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            seq_relationship_labels=None,
    ):

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        loss_fct = CrossEntropyLoss()

        sequence_output = outputs[0]

        sequence_output = self.predictions_dense(sequence_output)
        sequence_output = self.predictions_activation(sequence_output)
        sequence_output = self.predictions_LayerNorm(sequence_output)
        prediction_scores = self.predictions_decoder(sequence_output)

        if masked_lm_labels is not None:
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size)
                                      , masked_lm_labels.view(-1))

        pooled_output = outputs[1]
        seq_relationship_scores = self.seq_relationship(pooled_output)
        if seq_relationship_labels is not None:
            seq_relationship_loss = loss_fct(seq_relationship_scores.view(-1, 2), seq_relationship_labels.view(-1))

        loss = masked_lm_loss + seq_relationship_loss

        return loss


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr']  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss


class AlbertForTokenClassification(AlbertPreTrainedModel):

    def __init__(self, albert, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = albert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


config = AlbertConfig.from_json_file("D:/model_file/hfl_chinese-roberta-wwm-ext/config.json")
albert_pretrain = AlbertForPretrain(config)

checkpoint = torch.load("D:/model_file/hfl_chinese-roberta-wwm-ext/pretrain_checkpoint")
albert_pretrain.load_state_dict(checkpoint['model_state_dict'])

model_tokenclassification = AlbertForTokenClassification(albert_pretrain.albert, config)

from torch.optim import Adam
LEARNING_RATE = 0.0000003
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model_tokenclassification.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model_tokenclassification.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=LEARNING_RATE)


