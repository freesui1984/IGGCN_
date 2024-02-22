import os
import random
import numpy as np
from collections import Counter
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score, auc, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from .models.graph_clf import GraphClf
from .models.text_graph_clf import TextGraphRegression, TextGraphClf
from .models.eeg_graph_clf import EEGGraphRegression, EEGGraphClf
from .utils.text_data.vocab_utils import VocabModel
from .utils import constants as Constants
from .utils.generic_utils import to_cuda, create_mask
from .utils.constants import INF
from .utils.radam import RAdam


# 模型搭建要用到的方法以及初始化。为Model_Handler提供技术支持
class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    # 处理初始化基础网络的高级模型体系结构、保存、更新示例和预测示例。
    def __init__(self, config, train_set=None):
        self.config = config
        # network = net_module = GraphClf
        if self.config['model_name'] == 'GraphClf':
            self.net_module = GraphClf
        elif self.config['model_name'] == 'TextGraphRegression':
            self.net_module = TextGraphRegression
        elif self.config['model_name'] == 'TextGraphClf':
            self.net_module = TextGraphClf
        elif self.config['model_name'] == 'EEGGraphClf':
            self.net_module = EEGGraphClf
        else:
            raise RuntimeError('Unknown model_name: {}'.format(self.config['model_name']))
        print('[ Running {} model ]'.format(self.config['model_name']))
        # 文本数据
        if config['data_type'] == 'text':
            saved_vocab_file = os.path.join(config['data_dir'], '{}_seed{}.vocab'.format(config['dataset_name'], config.get('data_seed', 1234)))
            self.vocab_model = VocabModel.build(saved_vocab_file, train_set, self.config)
        # 回归（不用看）
        if config['task_type'] == 'regression':
            assert config['out_predictions']
            self.criterion = F.mse_loss
            self.score_func = r2_score
            self.metric_name = 'r2'
        # 分类
        elif config['task_type'] == 'classification':
            # self.criterion = FocalLoss(gamma=2, alpha=None, reduction='mean')
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = FocalLoss(gamma=2, alpha=[1 / 12912, 1 / 5054, 1 / 165, 1 / 561], reduction='mean')
            # self.criterion = F.nll_loss
            self.score_func = accuracy
            self.score_func2 = prf
            self.score_func3 = auc
            self.metric_name = 'acc'
            self.metric_name_precision = 'precision'
            self.metric_name_recall = 'recall'
            self.metric_name_f1 = 'f1'
            self.metric_name_Weight_F1 = 'Weight_F1'
            # self.metric_name_auc = 'auc'
        else:
            self.criterion = F.nll_loss
            self.score_func = None
            self.metric_name = None

        # 模型是否是预先训练好的
        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            # Building network.
            self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('#Parameters = {}\n'.format(num_params))
        self._init_optimizer()

    # 初始化保存的模型
    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['word_embed_dim', 'hidden_size', 'f_qem', 'f_pos', 'f_ner',
                      'word_dropout', 'rnn_dropout',
                      'ctx_graph_hops', 'ctx_graph_topk',
                      'score_unk_threshold', 'score_yes_threshold',
                      'score_no_threshold']

        # Load all saved fields.
        # 文件
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']
        # for k in _ARGUMENTS:
        #     if saved_params['config'][k] != self.config[k]:
        #         print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
        #         self.config[k] = saved_params['config'][k]

        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'])
            self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab)
        else:
            self.network = self.net_module(self.config)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    # 创建图结构
    def _init_new_network(self):
        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                               pretrained_vecs=self.vocab_model.word_vocab.embeddings)
            self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab)
        elif self.config['data_type'] == 'eeg':
            self.network = self.net_module(self.config)
        else:
            self.network = self.net_module(self.config)

    # 初始化优化器
    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adagrad':
            self.optimizer = optim.Adagrad(parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
                    patience=self.config['lr_patience'], verbose=True)

    # 文本数据集初始化嵌入，将文字转换为一串数字
    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        # 翻译过来的意思就是词嵌入，通俗来讲就是将文字转换为一串数字
        return nn.Embedding(vocab_size, embed_size, padding_idx=0,
                            _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    # 保存模型参数
    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    # 梯度渐变
    def clip_grad(self):
        # Clip gradients
        if self.config['grad_clipping']:
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # 根据类别设置权重
            alpha_factor = torch.tensor(self.alpha, device=input.device)[target]
            focal_loss = alpha_factor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def Focal_loss(output, labels):
    gamma = 0.1  # Focal Loss 的超参数，可以根据需要进行调整
    # 计算样本权重
    num_positive_samples = 1539  # 正类别样本数量
    num_negative_samples = 23384  # 负类别样本数量

    # 使用样本权重调整 Focal Loss
    positive_weight = 1 / num_positive_samples
    negative_weight = 1 / num_negative_samples

    # 计算 Focal Loss
    pt = F.softmax(output, dim=1)[:, 1]
    focal_loss = -((1 - pt) ** gamma) * torch.log(pt)

    # 对正类别和负类别应用不同的权重
    # loss = torch.sum(labels * focal_loss * positive_weight + (1 - labels) * focal_loss * negative_weight)

    loss = torch.sum(labels * focal_loss + (1 - labels) * focal_loss)


    # # 为每个类别创建权重张量
    # weights = torch.tensor([negative_weight, positive_weight], device=output.device)
    #
    # # 使用权重计算 CrossEntropy Loss
    # criterion = nn.CrossEntropyLoss(weight=weights)
    # loss = criterion(output, labels)

    return loss

# 准确率
def accuracy(labels, preds):
    correct = np.equal(preds, labels).astype(np.double)
    correct = correct.sum().item()
    return correct / len(labels)


# precision, recall, f1
def prf(labels, preds):
    scores_dict = {}
    # # 示例：根据样本数量分配权重
    # num_positive_samples = sum(labels)  # 正类别样本数量
    # num_negative_samples = len(labels) - num_positive_samples  # 负类别样本数量
    # weight_positive = 1.0 / num_positive_samples
    # weight_negative = 1.0 / num_negative_samples
    # sample_weights = [weight_positive if label == 1 else weight_negative for label in labels]
    #
    #
    # scores_dict['precision'] = precision_score(labels, preds, zero_division=0, sample_weight=sample_weights)
    # scores_dict['recall'] = recall_score(labels, preds, sample_weight=sample_weights)
    # scores_dict['f1'] = f1_score(labels, preds, average='macro')
    # scores_dict['Weighted_F1'] = f1_score(labels, preds, sample_weight=sample_weights)
    # from sklearn.metrics import roc_auc_score
    # scores_dict['auc'] = roc_auc_score(labels, preds)

    scores_dict['precision'] = precision_score(labels, preds, zero_division=0, average='weighted')
    scores_dict['recall'] = recall_score(labels, preds, average='weighted')
    scores_dict['f1'] = f1_score(labels, preds, average='weighted')
    scores_dict['Weighted_F1'] = scores_dict['f1']
    # from sklearn.metrics import roc_auc_score
    # scores_dict['auc'] = roc_auc_score(labels, preds)


    return scores_dict



# auc
def auc(labels, output):
    output = F.softmax(output, dim=1)
    preds = torch.argmax(output, dim=1)
    preds_auc = output[:, 1]

    # 假设 preds 和 labels 是位于 GPU 上的 PyTorch 张量
    preds_cpu = preds.cpu().detach().numpy()
    labels_cpu = labels.cpu().detach().numpy()

    # 使用 roc_curve 函数计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels_cpu, preds_cpu, pos_label=1)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels_cpu, preds_cpu)

    return auc
