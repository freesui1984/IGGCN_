# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import re
import random as rd
import io
import torch
import numpy as np
from scipy.sparse import *
from collections import Counter, defaultdict

from .network_data import data_utils as network_data_utils
from .uci_data import data_utils as uci_data_utils
from .text_data import data_utils as text_data_utils
from .eeg_data import eeg_data_utils as eeg_data_utils
from .timer import Timer
from . import padding_utils
from . import constants


def vectorize_input_text(batch, config, training=True, device=None):
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch.sent1_word)
    # batch_size = config['batch_size']

    context = torch.tensor(batch.sent1_word)
    context_lens = torch.LongTensor(batch.sent1_length)
    if config['task_type'] == 'regression':
        targets = torch.Tensor(batch.labels)
    elif config['task_type'] == 'classification':
        targets = torch.LongTensor(batch.labels)
    else:
        raise ValueError('Unknwon task_type: {}'.format(config['task_type']))

    if batch.has_sent2:
        context2 = torch.LongTensor(batch.sent2_word)
        context2_lens = torch.LongTensor(batch.sent2_length)

    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'context': context.to(device) if device else context,
                   'context_lens': context_lens.to(device) if device else context_lens,
                   'targets': targets.to(device) if device else targets}

        if batch.has_sent2:
            example['context2'] = context2.to(device) if device else context2
            example['context2_lens'] = context2_lens.to(device) if device else context2_lens
        return example


def vectorize_input_eeg(batch, config, training=True, device=None):
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    # batch_size = len(batch.sent1_word)
    batch_size = len(batch)
    context = []
    targets = []
    adj = []
    idx = []
    for i in range(len(batch)):
        context.append(batch[i][0])
        targets.append(batch[i][1])
        adj.append(batch[i][2])
        idx.append(batch[i][3])
    context = torch.FloatTensor(np.asarray(context))
    targets = torch.LongTensor(np.asarray(targets))
    adj = torch.FloatTensor(np.asarray(adj))
    idx = torch.LongTensor(np.asarray(idx))

    context_lens = torch.full([batch_size], config['channel'])

    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'context': context.to(device) if device else context,
                   'context_lens': context_lens.to(device) if device else context_lens,
                   'targets': targets.to(device) if device else targets,
                   'adj': adj.to(device) if device else adj,
                   'idx': idx.to(device) if device else idx}

        return example


def prepare_datasets(config):
    data = {}
    if config['data_type'] == 'eeg':
        train_set, dev_set, test_set = eeg_data_utils.load_data(config)
        print('# of training examples: {}'.format(len(train_set)))
        print('# of dev examples: {}'.format(len(dev_set)))
        print('# of testing examples: {}'.format(len(test_set)))
        data = {'train': train_set, 'dev': dev_set, 'test': test_set}

    else:
        raise ValueError('Unknown data_type: {}'.format(config['data_type']))
    return data


class EEGDataStream(object):
    def __init__(self, all_instances, config=None, isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']

        if isSort:
            all_instances = sorted(all_instances, key=lambda instance: [len(x) for x in instance[:-1]]) # the last element is label
        else:
            rd.shuffle(all_instances)
            rd.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_batch = all_instances[batch_start: batch_end]
            # cur_batch = InstanceBatch(cur_instances, config, word_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        # cur_pointer:当前是在哪个batch
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]


class DataStream(object):
    def __init__(self, all_instances, word_vocab, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        # 排序
        if isSort:
            all_instances = sorted(all_instances, key=lambda instance: [len(x) for x in instance[:-1]]) # the last element is label
        else:
            rd.shuffle(all_instances)
            rd.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = InstanceBatch(cur_instances, config, word_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        # cur_pointer:当前是在哪个batch
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]


class InstanceBatch(object):
    def __init__(self, instances, config, word_vocab):
        self.instances = instances
        self.batch_size = len(instances)
        if len(instances[0]) == 2:
            self.has_sent2 = False
        elif len(instances[0]) == 3:
            self.has_sent2 = True
        else:
            raise RuntimeError('{} elements per example, should be 2 or 3'.format(len(instances[0])))

        # Create word representation and length
        self.sent1_word = [] # [batch_size, sent1_len]
        self.sent1_length = [] # [batch_size]
        self.labels = [] # [batch_size]

        if self.has_sent2:
            self.sent2_word = [] # [batch_size, sent3_len]
            self.sent2_length = [] # [batch_size]

        for instance in self.instances:
            sent1_cut = instance[0][: config.get('max_seq_len', None)]
            self.sent1_word.append([word_vocab.getIndex(word) for word in sent1_cut])
            self.sent1_length.append(len(sent1_cut))
            if self.has_sent2:
                sent2_cut = instance[1][: config.get('max_seq_len', None)]
                self.sent2_word.append([word_vocab.getIndex(word) for word in sent2_cut])
                self.sent2_length.append(len(sent2_cut))
            self.labels.append(instance[-1])

        self.sent1_word = padding_utils.pad_2d_vals_no_size(self.sent1_word)
        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        if self.has_sent2:
            self.sent2_word = padding_utils.pad_2d_vals_no_size(self.sent2_word)
            self.sent2_length = np.array(self.sent2_length, dtype=np.int32)
