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
    if config['dataset_name'] == 'TUEP_EEG':
        file_path = os.path.join(config['data_dir'], config['feature_file_name'])
        index_file_path = os.path.join(config['data_dir'], config['index_file_name'])
        lable_path = os.path.join(config['data_dir'], config['label_file_name'])
        adj_path = os.path.join(config['data_dir'], config['adj_file_name'])
        train_set, dev_set, test_set = load_tuep_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, config.get('data_seed', 1234))
    elif config['dataset_name'] == 'TUAB_EEG':
        file_path = os.path.join(config['data_dir'], config['feature_file_name'])
        index_file_path = os.path.join(config['data_dir'], config['index_file_name'])
        lable_path = os.path.join(config['data_dir'], config['label_file_name'])
        adj_path = os.path.join(config['data_dir'], config['adj_file_name'])
        train_set, dev_set, test_set = load_tuab_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, config.get('data_seed', 1234))
    elif config['dataset_name'] == 'TUSZ_EEG':
        file_path = os.path.join(config['data_dir'], config['feature_file_name'])
        index_file_path = os.path.join(config['data_dir'], config['index_file_name'])
        lable_path = os.path.join(config['data_dir'], config['label_file_name'])
        adj_path = os.path.join(config['data_dir'], config['adj_file_name'])
        train_set, dev_set, test_set = load_tusz_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, config.get('data_seed', 1234))
    # elif config['dataset_name'] == 'TUSZ_EEG':
    #     # file_path = [config['data_dir'] + '/' + file_name for file_name in config['feature_file_name']]
    #     # index_file_path = [config['data_dir'] + '/' + file_name for file_name in config['index_file_name']]
    #     # label_path = [config['data_dir'] + '/' + file_name for file_name in config['label_file_name']]
    #     # adj_path = [config['data_dir'] + '/' + file_name for file_name in config['adj_file_name']]
    #     file_path = [os.path.join(config['data_dir'], file_name) for file_name in config['feature_file_name']]
    #     index_file_path = [os.path.join(config['data_dir'], file_name) for file_name in config['index_file_name']]
    #     label_path = [os.path.join(config['data_dir'], file_name) for file_name in config['label_file_name']]
    #     adj_path = [os.path.join(config['data_dir'], file_name) for file_name in config['adj_file_name']]
    #     train_set, dev_set, test_set = load_tusz_eeg_data(file_path, index_file_path, label_path, adj_path, data_split, config.get('data_seed', 1234))
    else:
        raise ValueError('Unknown dataset_name: {}'.format(config['dataset_name']))
    return train_set, dev_set, test_set


def load_tuep_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, seed):

    # 第一步划分数据集（train_dev, test）
    train_dev_ratio, test_ratio = data_split
    assert train_dev_ratio + test_ratio == 1
    # 数据索引
    all_data = pd.read_csv(index_file_path, dtype={"patient_ID": str})
    all_data_index = list(all_data.index)
    len_data = len(all_data_index)
    # 数据长度
    n_train_dev = int(len_data * train_dev_ratio)
    n_test = len_data - n_train_dev
    import random
    train_dev_index = random.sample(list(all_data.index), n_train_dev)
    train_dev_patient_ID = list(all_data.iloc[train_dev_index, 0])
    # test_index = random.sample(list(all_data.index), n_test)
    test_index = [item for item in all_data_index if item not in train_dev_index]
    test_patient_ID = list(all_data.iloc[test_index, 0])

    # 根据索引加载数据
    train_dev_instances, train_dev_labels, train_dev_adj = data_load_helper(file_path, lable_path, adj_path, train_dev_index)
    test_instances, test_labels, test_adj = data_load_helper(file_path, lable_path, adj_path, test_index)

    # 标签01化
    le = preprocessing.LabelEncoder()
    le.fit(train_dev_labels.tolist() + test_labels.tolist())
    nclass = len(list(le.classes_))
    print('[# of classes: {}] '.format(nclass))

    # data和label合并
    train_dev_labels = le.transform(train_dev_labels)
    test_labels = le.transform(test_labels)
    train_dev_instances = list(zip(train_dev_instances, train_dev_labels, train_dev_adj, np.array(train_dev_index)))
    test_instances = list(zip(test_instances, test_labels, test_adj, np.array(test_index)))

    # 第二步划分数据集（train, dev, test）
    train_ratio, dev_ratio = data_split
    assert train_ratio + dev_ratio == 1
    n_train = int(len(train_dev_instances) * train_ratio)

    random = np.random.RandomState(seed)
    random.shuffle(train_dev_instances)

    train_instances = train_dev_instances[:n_train]
    dev_instances = train_dev_instances[n_train:]
    return train_instances, dev_instances, test_instances


def load_tuab_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, seed):

    # 第一步划分数据集（train_dev, test）
    train_dev_ratio, test_ratio = data_split
    assert train_dev_ratio + test_ratio == 1
    # 数据索引
    all_data = pd.read_csv(index_file_path, dtype={"patient_ID": str}, index_col=0)
    all_data_index = list(all_data.index)
    len_data = len(all_data_index)
    # 数据长度
    n_train_dev = int(len_data * train_dev_ratio)
    n_test = len_data - n_train_dev
    import random
    train_dev_index = random.sample(list(all_data.index), n_train_dev)
    test_index = [item for item in all_data_index if item not in train_dev_index]
    test_patient_ID = list(all_data.iloc[test_index, 0])

    # 根据索引加载数据
    train_dev_instances, train_dev_labels, train_dev_adj = data_load_helper(file_path, lable_path, adj_path, train_dev_index)
    test_instances, test_labels, test_adj = data_load_helper(file_path, lable_path, adj_path, test_index)

    # 标签01化
    le = preprocessing.LabelEncoder()
    le.fit(train_dev_labels.tolist() + test_labels.tolist())
    nclass = len(list(le.classes_))
    print('[# of classes: {}] '.format(nclass))

    # data和label合并
    train_dev_labels = le.transform(train_dev_labels)
    test_labels = le.transform(test_labels)
    train_dev_instances = list(zip(train_dev_instances, train_dev_labels, train_dev_adj, np.array(train_dev_index)))
    test_instances = list(zip(test_instances, test_labels, test_adj, np.array(test_index)))

    # 第二步划分数据集（train, dev, test）
    train_ratio, dev_ratio = data_split
    assert train_ratio + dev_ratio == 1
    n_train = int(len(train_dev_instances) * train_ratio)

    random = np.random.RandomState(seed)
    random.shuffle(train_dev_instances)

    train_instances = train_dev_instances[:n_train]
    dev_instances = train_dev_instances[n_train:]
    return train_instances, dev_instances, test_instances

def load_tusz_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, seed):

    # 第一步划分数据集（train_dev, test）
    train_dev_ratio, test_ratio = data_split
    assert train_dev_ratio + test_ratio == 1
    # 数据索引
    all_data = pd.read_csv(index_file_path, dtype={"patient_ID": str})

    # 获取四类样本的索引
    class_labels = ['cfsz', 'gnsz', 'absz', 'ctsz']
    class_indices = [all_data[all_data['label'] == label].index.tolist() for label in class_labels]

    # 分别划分四类样本的索引
    class_train_dev_indices = [random.sample(indices, int(len(indices) * train_dev_ratio)) for indices in class_indices]
    class_test_indices = [list(set(indices) - set(train_indices)) for indices, train_indices in
                         zip(class_indices, class_train_dev_indices)]

    # 合并四类样本的索引，得到最终的训练集索引和验证集索引
    train_dev_index = sum(class_train_dev_indices, [])
    test_index = sum(class_test_indices, [])

    # 根据索引加载数据
    train_dev_instances, train_dev_labels, train_dev_adj = data_load_helper(file_path, lable_path, adj_path, train_dev_index)
    test_instances, test_labels, test_adj = data_load_helper(file_path, lable_path, adj_path, test_index)

    # 标签编码
    le = preprocessing.LabelEncoder()

    # 设置类别映射
    class_mapping = {'cfsz': 0, 'gnsz': 1, 'absz': 2, 'ctsz': 3}

    # 将原始标签映射为数字
    train_dev_labels = np.vectorize(class_mapping.get)(train_dev_labels)
    test_labels = np.vectorize(class_mapping.get)(test_labels)

    # 使用 LabelEncoder 进行标签编码
    le.fit(list(class_mapping.values()))

    # 转换为编码后的数字标签
    train_dev_labels = le.transform(train_dev_labels)
    test_labels = le.transform(test_labels)
    train_dev_instances = list(zip(train_dev_instances, train_dev_labels, train_dev_adj, np.array(train_dev_index)))
    test_instances = list(zip(test_instances, test_labels, test_adj, np.array(test_index)))

    # 第二步划分数据集（train, dev, test）
    train_ratio, dev_ratio = data_split
    assert train_ratio + dev_ratio == 1


    train_index = []
    dev_index = []

    for class_label in class_labels:
        # 使用数字映射获取当前类别的标签
        current_class_label = class_mapping[class_label]
        # 获取当前类别的索引
        class_indices = [index for index, label in zip(train_dev_index, train_dev_labels) if label == current_class_label]

        # 分别划分当前类别样本的索引
        class_train_indices = random.sample(class_indices, int(len(class_indices) * train_ratio))
        class_dev_indices = list(set(class_indices) - set(class_train_indices))

        # 将实例和索引分别加入对应的集合中
        train_index.extend(class_train_indices)
        dev_index.extend(class_dev_indices)

    # 根据索引加载数据
    train_instances, train_labels, train_adj = data_load_helper(file_path, lable_path, adj_path, train_index)
    dev_instances, dev_labels, dev_adj = data_load_helper(file_path, lable_path, adj_path, dev_index)

    # 标签编码
    le = preprocessing.LabelEncoder()

    # 设置类别映射
    class_mapping = {'cfsz': 0, 'gnsz': 1, 'absz': 2, 'ctsz': 3}

    # 将原始标签映射为数字
    train_labels = np.vectorize(class_mapping.get)(train_labels)
    dev_labels = np.vectorize(class_mapping.get)(dev_labels)

    # 使用 LabelEncoder 进行标签编码
    le.fit(list(class_mapping.values()))

    # 转换为编码后的数字标签
    train_labels = le.transform(train_labels)
    dev_labels = le.transform(dev_labels)
    train_instances = list(zip(train_instances, train_labels, train_adj, np.array(train_index)))
    dev_instances = list(zip(dev_instances, dev_labels, dev_adj, np.array(dev_index)))

    # 随机打乱训练集、验证集和测试集
    random.shuffle(train_instances)
    random.shuffle(dev_instances)
    random.shuffle(test_instances)

    # # 只取各自 30% 返回
    # new_n_train = int(len(train_instances) * 0.1)
    # new_n_dev = int(len(dev_instances) * 0.1)
    # new_n_test = int(len(test_instances) * 0.1)
    #
    # random.shuffle(train_instances)
    # random.shuffle(dev_instances)
    # random.shuffle(test_instances)
    #
    # train_instances = train_instances[:new_n_train]
    # dev_instances = dev_instances[:new_n_dev]
    # test_instances = test_instances[:new_n_test]

    return train_instances, dev_instances, test_instances

# def load_tusz_eeg_data(file_path, index_file_path, lable_path, adj_path, data_split, seed):
#
#     # 第一步划分数据集（train_dev, test）
#     train_ratio, dev_ratio = data_split
#     assert train_ratio + dev_ratio == 1
#     # 数据索引
#     all_data = pd.read_csv(index_file_path[0], dtype={"patient_ID": str})
#     import random
#
#     # 获取两类样本的索引
#     positive_indices = all_data[all_data['label'] == 'seiz'].index.tolist()
#     negative_indices = all_data[all_data['label'] == 'no_seiz'].index.tolist()
#
#     # 分别划分正负样本的索引
#     n_positive_train = int(len(positive_indices) * train_ratio)
#     n_positive_dev = len(positive_indices) - n_positive_train
#
#     n_negative_train = int(len(negative_indices) * train_ratio)
#     n_negative_dev = len(negative_indices) - n_negative_train
#
#     # 随机选择正负样本的索引
#     positive_train_index = random.sample(list(positive_indices), n_positive_train)
#     positive_dev_index = random.sample(list(positive_indices), n_positive_dev)
#
#     negative_train_index = random.sample(list(negative_indices), n_negative_train)
#     negative_dev_index = random.sample(list(negative_indices), n_negative_dev)
#
#     # 合并正负样本的索引，得到最终的训练集索引和验证集索引
#     train_index = positive_train_index + negative_train_index
#     dev_index = positive_dev_index + negative_dev_index
#
#     # 根据索引加载数据
#     train_instances, train_labels, train_adj = data_load_helper(file_path[0], lable_path[0], adj_path[0], train_index)
#     dev_instances, dev_labels, dev_adj = data_load_helper(file_path[0], lable_path[0], adj_path[0], dev_index)
#
#     # 标签01化
#     le = preprocessing.LabelEncoder()
#     le.fit(train_labels.tolist() + dev_labels.tolist())
#     nclass = len(list(le.classes_))
#     print('[# of classes: {}] '.format(nclass))
#
#     # data和label合并
#     train_labels = le.transform(train_labels)
#     dev_labels = le.transform(dev_labels)
#     train_instances = list(zip(train_instances, train_labels, train_adj, np.array(train_index)))
#     dev_instances = list(zip(dev_instances, dev_labels, dev_adj, np.array(dev_index)))
#     random.shuffle(train_instances)
#     random.shuffle(dev_instances)
#
#     # 测试集
#     test_data = pd.read_csv(index_file_path[2], dtype={"patient_ID": str})
#     test_data_index = list(test_data.index)
#     # 根据索引加载数据
#     test_instances, test_labels, test_adj = data_load_helper(file_path[2], lable_path[2], adj_path[2], test_data_index)
#     # 标签01化
#     le = preprocessing.LabelEncoder()
#     le.fit(test_labels.tolist())
#     # data和label合并
#     test_labels = le.transform(test_labels)
#     test_instances = list(zip(test_instances, test_labels, test_adj, np.array(test_data_index)))
#     random.shuffle(test_instances)
#
#
#     return train_instances, dev_instances, test_instances