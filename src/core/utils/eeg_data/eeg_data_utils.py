import os
import re
import codecs
import string
from joblib import load
import random
import pandas as pd
from collections import defaultdict
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from sklearn import preprocessing
# https://github.com/tkipf/gcn/blob/master/gcn/utils.py
import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import torch
from ..generic_utils import *
from ..constants import VERY_SMALL_NUMBER


def data_load_helper(data_dir, lable_dir, adj_dir, index):

    data = np.load(data_dir, allow_pickle=True)
    labels = np.load(lable_dir, allow_pickle=True)
    adj = np.load(adj_dir, allow_pickle=True)
    all_instances = data[index]
    all_labels = labels[index]
    all_adjs = adj[index]

    return all_instances, all_labels, all_adjs


def load_data(config):
    data_split = [float(x) for x in config['data_split_ratio'].replace(' ', '').split(',')]
    if config['dataset_name'] == 'TUSZ_EEG':
        file_path = os.path.join(config['data_dir'], config['feature_file_name'])
        index_file_path = os.path.join(config['data_dir'], config['index_file_name'])
        lable_path = os.path.join(config['data_dir'], config['label_file_name'])
        adj_path = os.path.join(config['data_dir'], config['adj_file_name'])
        train_set, dev_set, test_set = load_tusz_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, config.get('data_seed', 1234))
    else:
        raise ValueError('Unknown dataset_name: {}'.format(config['dataset_name']))
    return train_set, dev_set, test_set


def load_tusz_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, seed):

    # Step 1 Divide the dataset (train_dev, test)
    train_dev_ratio, test_ratio = data_split
    assert train_dev_ratio + test_ratio == 1
    all_data = pd.read_csv(index_file_path, dtype={"patient_ID": str})

    # Get an index of four types of samples
    class_labels = ['cfsz', 'gnsz', 'absz', 'ctsz']
    class_indices = [all_data[all_data['label'] == label].index.tolist() for label in class_labels]


    class_train_dev_indices = [random.sample(indices, int(len(indices) * train_dev_ratio)) for indices in class_indices]
    class_test_indices = [list(set(indices) - set(train_indices)) for indices, train_indices in
                         zip(class_indices, class_train_dev_indices)]
    train_dev_index = sum(class_train_dev_indices, [])
    test_index = sum(class_test_indices, [])

    # Load data according to index
    train_dev_instances, train_dev_labels, train_dev_adj = data_load_helper(file_path, lable_path, adj_path, train_dev_index)
    test_instances, test_labels, test_adj = data_load_helper(file_path, lable_path, adj_path, test_index)

    # Label coding
    le = preprocessing.LabelEncoder()
    class_mapping = {'cfsz': 0, 'gnsz': 1, 'absz': 2, 'ctsz': 3}
    train_dev_labels = np.vectorize(class_mapping.get)(train_dev_labels)
    test_labels = np.vectorize(class_mapping.get)(test_labels)
    le.fit(list(class_mapping.values()))

    train_dev_labels = le.transform(train_dev_labels)
    test_labels = le.transform(test_labels)
    train_dev_instances = list(zip(train_dev_instances, train_dev_labels, train_dev_adj, np.array(train_dev_index)))
    test_instances = list(zip(test_instances, test_labels, test_adj, np.array(test_index)))

    # Step 2 Divide the data set (train, dev, test)
    train_ratio, dev_ratio = data_split
    assert train_ratio + dev_ratio == 1


    train_index = []
    dev_index = []

    for class_label in class_labels:
        current_class_label = class_mapping[class_label]
        class_indices = [index for index, label in zip(train_dev_index, train_dev_labels) if label == current_class_label]

        class_train_indices = random.sample(class_indices, int(len(class_indices) * train_ratio))
        class_dev_indices = list(set(class_indices) - set(class_train_indices))

        train_index.extend(class_train_indices)
        dev_index.extend(class_dev_indices)

    train_instances, train_labels, train_adj = data_load_helper(file_path, lable_path, adj_path, train_index)
    dev_instances, dev_labels, dev_adj = data_load_helper(file_path, lable_path, adj_path, dev_index)

    le = preprocessing.LabelEncoder()

    class_mapping = {'cfsz': 0, 'gnsz': 1, 'absz': 2, 'ctsz': 3}

    train_labels = np.vectorize(class_mapping.get)(train_labels)
    dev_labels = np.vectorize(class_mapping.get)(dev_labels)

    le.fit(list(class_mapping.values()))

    train_labels = le.transform(train_labels)
    dev_labels = le.transform(dev_labels)
    train_instances = list(zip(train_instances, train_labels, train_adj, np.array(train_index)))
    dev_instances = list(zip(dev_instances, dev_labels, dev_adj, np.array(dev_index)))

    # Shuffle
    random.shuffle(train_instances)
    random.shuffle(dev_instances)
    random.shuffle(test_instances)


    return train_instances, dev_instances, test_instances
