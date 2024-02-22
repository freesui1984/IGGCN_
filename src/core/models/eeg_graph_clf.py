import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from dotted_dict import DottedDict

from ..layers.graphlearn import GraphLearner, get_binarized_kneighbors_graph
from ..layers.scalable_graphlearn import AnchorGraphLearner
from ..layers.anchor import AnchorGCN
from ..layers.common import dropout, EncoderRNN
from ..layers.gnn import GCN, GAT, GGNN
from ..compare_model.lstm import LSTMModel
from ..compare_model.cnnlstm import CNN_LSTM
from ..compare_model.densecnn import DenseCNN
from ..utils.generic_utils import to_cuda, create_mask, batch_normalize_adj, normalize_adj
from ..utils.constants import VERY_SMALL_NUMBER


# 这个没改-------------------------
class EEGGraphRegression(nn.Module):
    def __init__(self, config, w_embedding, word_vocab):
        super(EEGGraphRegression, self).__init__()
        self.config = config
        self.name = 'TextGraphRegression'
        self.device = config['device']

        # Shape
        word_embed_dim = config['word_embed_dim']
        hidden_size = config['hidden_size']

        # Dropout
        self.dropout = config['dropout']
        self.word_dropout = config.get('word_dropout', config['dropout'])
        self.rnn_dropout = config.get('rnn_dropout', config['dropout'])

        # Graph
        self.graph_learn = config['graph_learn']
        self.graph_metric_type = config['graph_metric_type']
        self.graph_module = config['graph_module']
        self.graph_skip_conn = config['graph_skip_conn']
        self.graph_include_self = config.get('graph_include_self', True)

        # Text
        self.word_embed = w_embedding
        if config['fix_vocab_embed']:
            print('[ Fix word embeddings ]')
            for param in self.word_embed.parameters():
                param.requires_grad = False

        self.ctx_rnn_encoder = EncoderRNN(word_embed_dim, hidden_size, bidirectional=True, num_layers=1, rnn_type='lstm',
                              rnn_dropout=self.rnn_dropout, device=self.device)

        self.linear_out = nn.Linear(hidden_size, 1, bias=False)
        self.scalable_run = config.get('scalable_run', False)

        if not config.get('no_gnn', False):
            print('[ Using TextGNN ]')
            if self.graph_module == 'gcn':
                gcn_module = AnchorGCN if self.scalable_run else GCN
                self.encoder = gcn_module(nfeat=hidden_size,
                                    nhid=hidden_size,
                                    nclass=hidden_size,
                                    graph_hops=config.get('graph_hops', 2),
                                    dropout=self.dropout,
                                    batch_norm=config.get('batch_norm', False))

            else:
                raise RuntimeError('Unknown graph_module: {}'.format(self.graph_module))


            if self.graph_learn:
                graph_learn_fun = AnchorGraphLearner if self.scalable_run else GraphLearner
                self.graph_learner = graph_learn_fun(word_embed_dim, config['graph_learn_hidden_size'],
                                                topk=config['graph_learn_topk'],
                                                epsilon=config['graph_learn_epsilon'],
                                                num_pers=config['graph_learn_num_pers'],
                                                metric_type=config['graph_metric_type'],
                                                device=self.device)


                self.graph_learner2 = graph_learn_fun(hidden_size,
                                                config.get('graph_learn_hidden_size2', config['graph_learn_hidden_size']),
                                                topk=config.get('graph_learn_topk2', config['graph_learn_topk']),
                                                epsilon=config.get('graph_learn_epsilon2', config['graph_learn_epsilon']),
                                                num_pers=config['graph_learn_num_pers'],
                                                metric_type=config['graph_metric_type'],
                                                device=self.device)

                print('[ Graph Learner ]')

                if config['graph_learn_regularization']:
                  print('[ Graph Regularization]')
            else:
                self.graph_learner = None
                self.graph_learner2 = None

        else:
            print('[ Using RNN ]')

    def compute_no_gnn_output(self, context, context_lens):
        raw_context_vec = self.word_embed(context)
        raw_context_vec = dropout(raw_context_vec, self.word_dropout, shared_axes=[-2], training=self.training)

        # Shape: [batch_size, hidden_size]
        context_vec = self.ctx_rnn_encoder(raw_context_vec, context_lens)[1][0].squeeze(0)
        output = self.linear_out(context_vec).squeeze(-1)
        return torch.sigmoid(output)

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, node_mask=None, anchor_mask=None, graph_include_self=False, init_adj=None, anchor_features=None):
        if self.graph_learn:
            if self.scalable_run:
                node_anchor_adj = graph_learner(node_features, anchor_features, node_mask, anchor_mask)
                return node_anchor_adj

            else:
                raw_adj = graph_learner(node_features, node_mask)

                if self.graph_metric_type in ('kernel', 'weighted_cosine'):
                    assert raw_adj.min().item() >= 0
                    adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
                elif self.graph_metric_type == 'cosine':
                    adj = (raw_adj > 0).float()
                    adj = normalize_adj(adj)
                else:
                    adj = torch.softmax(raw_adj, dim=-1)

                if graph_skip_conn in (0, None):
                    if graph_include_self:
                        adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
                else:
                    adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

                return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj

    def compute_output(self, node_vec, node_mask=None):
        graph_vec = self.graph_maxpool(node_vec.transpose(-1, -2), node_mask=node_mask)
        output = self.linear_out(graph_vec).squeeze(-1)
        return torch.sigmoid(output)

    def prepare_init_graph(self, context, context_lens):
        context_mask = create_mask(context_lens, context.size(-1), device=self.device)
        # Shape: [batch_size, max_length, word_embed_dim]
        raw_context_vec = self.word_embed(context)
        raw_context_vec = dropout(raw_context_vec, self.word_dropout, shared_axes=[-2], training=self.training)

        # Shape: [batch_size, max_length, hidden_size]
        context_vec = self.ctx_rnn_encoder(raw_context_vec, context_lens)[0].transpose(0, 1)

        init_adj = self.compute_init_adj(raw_context_vec.detach(), self.config['input_graph_knn_size'], mask=context_mask)
        return raw_context_vec, context_vec, context_mask, init_adj

    def graph_maxpool(self, node_vec, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_nodes)
        graph_embedding = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(-1)
        return graph_embedding

    def compute_init_adj(self, features, knn_size, mask=None):
        adj = get_binarized_kneighbors_graph(features, knn_size, mask=mask, device=self.device)
        adj_norm = batch_normalize_adj(adj, mask=mask)
        return adj_norm


# 针对EEG数据的模型进行更改
class EEGGraphClf(nn.Module):
    def __init__(self, config):
        super(EEGGraphClf, self).__init__()
        self.config = config
        self.name = 'EEGGraphClf'
        self.device = config['device']

        # Shape
        embed_dim = config['eeg_embed_dim']
        hidden_size = config['hidden_size']
        nclass = config['num_class']

        # Dropout
        self.dropout = config['dropout']
        self.word_dropout = config.get('word_dropout', config['dropout'])
        self.rnn_dropout = config.get('rnn_dropout', config['dropout'])

        # Graph
        self.graph_learn = config['graph_learn']
        self.graph_metric_type = config['graph_metric_type']
        self.graph_module = config['graph_module']
        self.graph_skip_conn = config['graph_skip_conn']
        self.graph_include_self = config.get('graph_include_self', True)

        # GGNN
        state_dim = hidden_size
        n_node = config['node_num']
        n_edge_types = 1
        annotation_dim = config['eeg_embed_dim']
        n_steps = config['min_word_freq']

        num_nodes = config['node_num']
        if self.graph_module == 'lstm':
            rnn_units = config['rnn_units']
            num_rnn_layers = config['num_rnn_layers']
        input_dim = config['eeg_embed_dim']



        self.ctx_rnn_encoder = EncoderRNN(embed_dim, hidden_size, bidirectional=True, num_layers=1, rnn_type='lstm'
                                          , rnn_dropout=self.rnn_dropout, device=self.device)

        self.linear_out = nn.Linear(hidden_size, nclass, bias=False)

        self.scalable_run = config.get('scalable_run', False)

        if not config.get('no_gnn', False):
            print('[ Using EEGGNN ]')
            # self.linear_max = nn.Linear(hidden_size, nclass, bias=False)

            if self.graph_module == 'gcn':
                gcn_module = AnchorGCN if self.scalable_run else GCN
                self.encoder = gcn_module(nfeat=hidden_size,
                                    nhid=hidden_size,
                                    nclass=hidden_size,
                                    graph_hops=config.get('graph_hops', 2),
                                    dropout=self.dropout,
                                    batch_norm=config.get('batch_norm', False))

            elif self.graph_module == 'ggnn':
                print('[ Using GGNN ]')
                gcn_module = GGNN
                self.encoder = gcn_module(state_dim=hidden_size,
                                          n_edge_types=n_edge_types,
                                          n_steps=n_steps,
                                          nclass=nclass)
            elif self.graph_module == 'lstm':
                print('[ Using LSTM ]')
                gcn_module = LSTMModel
                self.encoder = gcn_module(num_nodes=num_nodes,
                                          rnn_units=rnn_units,
                                          num_rnn_layers=num_rnn_layers,
                                          input_dim=input_dim,
                                          num_classes=nclass,
                                          dropout=self.dropout)

            elif self.graph_module == 'cnnlstm':
                print('[ Using CNNLSTM ]')
                gcn_module = CNN_LSTM
                self.encoder = gcn_module(num_classes=nclass)

            elif self.graph_module == 'densecnn':
                print('[ Using DenseCNN ]')
                gcn_module = DenseCNN
                with open("E:\\hy\\IDGL-master\\src\\core\\compare_model\\dense_inception\\params.json", "r") as f:
                    params = json.load(f)
                params = DottedDict(params)
                data_shape = (n_steps * 200, num_nodes)
                self.encoder = gcn_module(params=params,
                                          data_shape=data_shape,
                                          num_classes=nclass)

            else:
                raise RuntimeError('Unknown graph_module: {}'.format(self.graph_module))

            if self.graph_learn:
                graph_learn_fun = AnchorGraphLearner if self.scalable_run else GraphLearner
                self.graph_learner = graph_learn_fun(config, embed_dim, config['graph_learn_hidden_size'],
                                                topk=config['graph_learn_topk'],
                                                epsilon=config['graph_learn_epsilon'],
                                                num_pers=config['graph_learn_num_pers'],
                                                metric_type=config['graph_metric_type'],
                                                device=self.device)

                self.graph_learner2 = graph_learn_fun(config, hidden_size,
                                                config.get('graph_learn_hidden_size2', config['graph_learn_hidden_size']),
                                                topk=config.get('graph_learn_topk2', config['graph_learn_topk']),
                                                epsilon=config.get('graph_learn_epsilon2', config['graph_learn_epsilon']),
                                                num_pers=config['graph_learn_num_pers'],
                                                metric_type=config['graph_metric_type'],
                                                device=self.device)

                print('[ Graph Learner ]')

                if config['graph_learn_regularization']:
                  print('[ Graph Regularization]')
            else:
                self.graph_learner = None
                self.graph_learner2 = None

        else:
            print('[ Using RNN ]')

    def compute_no_gnn_output(self, context, context_lens):
        raw_context_vec = self.word_embed(context)
        raw_context_vec = dropout(raw_context_vec, self.word_dropout, shared_axes=[-2], training=self.training)

        # Shape: [batch_size, hidden_size]
        context_vec = self.ctx_rnn_encoder(raw_context_vec, context_lens)[1][0].squeeze(0)
        output = self.linear_out(context_vec)
        output = F.log_softmax(output, dim=-1)
        return output

    # 学习图结构
    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, node_mask=None, anchor_mask=None, graph_include_self=False, init_adj=None, anchor_features=None):
        if self.graph_learn:
            if self.scalable_run:
                node_anchor_adj = graph_learner(node_features, anchor_features, node_mask, anchor_mask)
                return node_anchor_adj
            # 主
            else:
                # raw_adj:返回的注意力分数
                raw_adj = graph_learner(node_features, init_adj, node_mask)

                if self.graph_metric_type in ('kernel', 'weighted_cosine'):
                    assert raw_adj.min().item() >= 0
                    adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
                elif self.graph_metric_type == 'cosine':
                    adj = (raw_adj > 0).float()
                    adj = normalize_adj(adj)
                else:
                    adj = torch.softmax(raw_adj, dim=-1)

                if graph_skip_conn in (0, None):
                    if graph_include_self:
                        adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
                else:
                    adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

                return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj

    def compute_output(self, node_vec, node_mask=None):
        graph_vec = self.graph_maxpool(node_vec.transpose(-1, -2), node_mask=node_mask)
        output = self.linear_out(graph_vec)
        output = F.log_softmax(output, dim=-1)
        return output

    # 准备初始的图结构
    def prepare_init_graph(self, context, context_lens, adj):

        # 考虑到每个句子的单词数量不一样，才要加上mask
        # context_mask = create_mask(context_lens, self.config['channel'], device=self.device)
        # Shape: [batch_size, max_length, word_embed_dim]
        # raw_context_vec = self.word_embed(context)
        raw_context_vec = context.reshape(len(context_lens), self.config['channel'], self.config['eeg_embed_dim'])
        raw_context_vec = dropout(raw_context_vec, self.word_dropout, shared_axes=[-2], training=self.training)

        # Shape: [batch_size, node_num, embed_dim]
        # 下面这条有待考虑
        context_vec = self.ctx_rnn_encoder(raw_context_vec, context_lens)[0].transpose(0, 1)
        # context_vec = context.reshape(self.config['batch_size'], len(context_lens), self.config['eeg_embed_dim'])
        # context_vec = raw_context_vec

        # init_adj = self.compute_init_adj(raw_context_vec.detach(), self.config['input_graph_knn_size'], mask=context_mask)
        init_adj = adj.reshape(len(context_lens), self.config['channel'], self.config['channel'])
        return raw_context_vec, context_vec, init_adj

    def graph_maxpool(self, node_vec, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_nodes)
        graph_embedding = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(-1)
        return graph_embedding

    # 创建初始图结构
    def compute_init_adj(self, features, knn_size, mask=None):
        adj = get_binarized_kneighbors_graph(features, knn_size, mask=mask, device=self.device)
        adj_norm = batch_normalize_adj(adj, mask=mask)
        return adj_norm
