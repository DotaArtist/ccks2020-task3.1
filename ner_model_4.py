# coding=utf-8
"""ner_model_4"""
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from module import feedforward
from module import positional_encoding
from module import multihead_attention


class Model4(object):
    def __init__(self, is_training=True, num_tags=13,
                 learning_rate=0.0001,
                 embedding_size=256,
                 sequence_length_val=1500,
                 keep_prob=0.9,
                 fc_hidden_num=200,
                 bilstm_hidden_num=100,
                 num_blocks=6,
                 num_headers=8,
                 encoder_hidden_dim=256,
                 feedfordward_hidden_dim=2048):

        self.sequence_length_val = sequence_length_val
        self.is_training = is_training
        self.num_tags = num_tags
        self.embedding_size = embedding_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.bilstm_hidden_num = bilstm_hidden_num
        self.fc_hidden_num = fc_hidden_num

        self.num_blocks = num_blocks
        self.num_headers = num_headers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.feedfordward_hidden_dim = feedfordward_hidden_dim
        self.dropout_rate = 1 - keep_prob

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.sequence_length_val, self.embedding_size], name='input_x')
        self.input_y = tf.placeholder(tf.int64, shape=[None, None], name='input_y')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.transition_params = None

        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.decode_tags = self.predict_label()

    def inference(self):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc = self.input_x
            enc *= self.encoder_hidden_dim**0.5  # scale

            enc += positional_encoding(enc, self.sequence_length_val)
            enc = tf.layers.dropout(enc, self.dropout_rate, training=self.is_training)

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_headers,
                                              dropout_rate=self.dropout_rate,
                                              training=self.is_training,
                                              causality=False)
                    enc = feedforward(enc, num_units=[self.feedfordward_hidden_dim, self.encoder_hidden_dim])  # bs * sl * ed
        memory = enc

        with tf.variable_scope('bilstm_layer', reuse=tf.AUTO_REUSE):
            cell_fw = LSTMCell(self.bilstm_hidden_num)
            cell_bw = LSTMCell(self.bilstm_hidden_num)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=memory,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            bilstm_layer_output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            bilstm_layer_output = tf.nn.dropout(bilstm_layer_output, self.keep_prob)

        with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(shape=[self.bilstm_hidden_num * 2, self.num_tags],
                                      initializer=tf.random_normal_initializer(), name="w",
                                      trainable=self.is_training)
            biases = tf.get_variable(shape=[self.num_tags],
                                     initializer=tf.random_normal_initializer(), name="b",
                                     trainable=self.is_training)
            s = tf.shape(bilstm_layer_output)
            bilstm_layer_output = tf.reshape(bilstm_layer_output, [-1, self.bilstm_hidden_num * 2])

            fc_output = tf.nn.xw_plus_b(bilstm_layer_output, weights, biases)
            logits = tf.reshape(fc_output, [-1, s[1], self.num_tags])

        return logits

    def loss(self,):
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                    tag_indices=self.input_y,
                                                                    sequence_lengths=self.sequence_lengths)
        crf_loss = -tf.reduce_mean(log_likelihood)
        return crf_loss

    def predict_label(self):
        decode_tags, best_score = tf.contrib.crf.crf_decode(potentials=self.logits,
                                                            transition_params=self.transition_params,
                                                            sequence_length=self.sequence_lengths
                                                            )
        self.decode_tags = decode_tags
        return decode_tags

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
        return train_op
