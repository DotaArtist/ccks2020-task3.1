
import argparse
import os
import numpy as np

from ..bilm.training import train, load_options_latest_checkpoint, load_vocab
from ..bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, None)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768648884

    options = {
     'bidirectional': True,

     # 'char_cnn': {'activation': 'relu',
     #  'embedding': {'dim': 16},
     #  'filters': [[1, 32],
     #   [2, 32],
     #   [3, 64],
     #   [4, 128],
     #   [5, 256],
     #   [6, 512],
     #   [7, 1024]],
     #  'max_characters_per_token': 50,
     #  'n_characters': 261,
     #  'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 1024,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 256,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 1024,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../try', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', default='../data/vocab.txt',  help='Vocabulary file')
    parser.add_argument('--train_prefix', default='../data/corpus.txt', help='Prefix for train files')

    args = parser.parse_args()
    main(args)
