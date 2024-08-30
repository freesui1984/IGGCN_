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
        train_file_path = os.path.join(config['data_dir'], config['train_feature_file_name'])
        train_index_file_path = os.path.join(config['data_dir'], config['train_index_file_name'])
        train_lable_path = os.path.join(config['data_dir'], config['train_label_file_name'])
        train_adj_path = os.path.join(config['data_dir'], config['train_adj_file_name'])

        dev_file_path = os.path.join(config['data_dir'], config['dev_feature_file_name'])
        dev_index_file_path = os.path.join(config['data_dir'], config['dev_index_file_name'])
        dev_lable_path = os.path.join(config['data_dir'], config['dev_label_file_name'])
        dev_adj_path = os.path.join(config['data_dir'], config['dev_adj_file_name'])

        test_file_path = os.path.join(config['data_dir'], config['test_feature_file_name'])
        test_index_file_path = os.path.join(config['data_dir'], config['test_index_file_name'])
        test_lable_path = os.path.join(config['data_dir'], config['test_label_file_name'])
        test_adj_path = os.path.join(config['data_dir'], config['test_adj_file_name'])

        train_set, dev_set, test_set = load_tusz_eeg_data(train_file_path, train_index_file_path, train_lable_path,
                                                      train_adj_path, dev_file_path, dev_index_file_path,
                                                      dev_lable_path,
                                                      dev_adj_path, test_file_path, test_index_file_path,
                                                      test_lable_path,
                                                      test_adj_path, data_split, config.get('data_seed', 1234))
    else:
        raise ValueError('Unknown dataset_name: {}'.format(config['dataset_name']))
    return train_set, dev_set, test_set


def load_tusz_eeg_data(train_file_path, train_index_file_path, train_lable_path, train_adj_path, dev_file_path, dev_index_file_path, dev_lable_path, dev_adj_path, test_file_path, test_index_file_path, test_lable_path, test_adj_path, data_split, seed):

    train_data = pd.read_csv(train_index_file_path, dtype={"patient_ID": str})
    dev_data = pd.read_csv(dev_index_file_path, dtype={"patient_ID": str})
    test_data = pd.read_csv(test_index_file_path, dtype={"patient_ID": str})

    class_labels = ['cfsz', 'gnsz', 'absz', 'ctsz']
    class_train_indices = [train_data[train_data['label'] == label].index.tolist() for label in class_labels]
    class_dev_indices = [dev_data[dev_data['label'] == label].index.tolist() for label in class_labels]
    class_test_indices = [test_data[test_data['label'] == label].index.tolist() for label in class_labels]

    train_index = sum(class_train_indices, [])
    dev_index = sum(class_dev_indices, [])
    test_index = sum(class_test_indices, [])

    train_instances, train_labels, train_adj = data_load_helper(train_file_path, train_lable_path, train_adj_path, train_index)
    dev_instances, dev_labels, dev_adj = data_load_helper(dev_file_path, dev_lable_path, dev_adj_path, dev_index)
    test_instances, test_labels, test_adj = data_load_helper(test_file_path, test_lable_path, test_adj_path, test_index)

    le = preprocessing.LabelEncoder()

    class_mapping = {'cfsz': 0, 'gnsz': 1, 'absz': 2, 'ctsz': 3}

    train_labels = np.vectorize(class_mapping.get)(train_labels)
    dev_labels = np.vectorize(class_mapping.get)(dev_labels)
    test_labels = np.vectorize(class_mapping.get)(test_labels)

    le.fit(list(class_mapping.values()))

    train_labels = le.transform(train_labels)
    dev_labels = le.transform(dev_labels)
    test_labels = le.transform(test_labels)
    train_instances = list(zip(train_instances, train_labels, train_adj, np.array(train_index)))
    dev_instances = list(zip(dev_instances, dev_labels, dev_adj, np.array(dev_index)))
    test_instances = list(zip(test_instances, test_labels, test_adj, np.array(test_index)))


    random.shuffle(train_instances)
    random.shuffle(dev_instances)
    random.shuffle(test_instances)


    return train_instances, dev_instances, test_instances