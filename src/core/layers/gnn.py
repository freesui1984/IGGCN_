import torch
import torch.nn as nn
import torch.nn.functional as F

# from .graphlearn import GraphLearner
from ..utils.generic_utils import to_cuda
from ..utils.constants import VERY_SMALL_NUMBER, INF


# 定义多种GNN模型


class GraphSAGE(nn.Module):
    """https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py"""
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        from dgl.nn.pytorch.conv import SAGEConv

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class EEGGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(EEGGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.w = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        # Xavier均匀分布初始化
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，这里有一个gain，增益的大小是依据激活函数类型来设定

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # torch.mm矩阵相乘
        Wh = torch.matmul(h, self.w)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # torch.matmul矩阵乘法，输入可以是高维；squeeze可以将维度为1的那个维度去掉，当输入值中
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 存在dim时，则只有当dim对应的维度为1时会实现降维
        # eij = a([Whi||Whj]),j属于Ni

        # 范围维度和e一样的全1的矩阵
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention
        # h_prime = torch.matmul(attention, Wh)
        #
        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # number of nodes
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # epeat_interleave()：在原有的tensor上，按每一个tensor复制。
        # repeat()：根据原有的tensor复制n个，然后拼接在一起。
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # dim=0表示按行拼接，1表示按列拼接
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)
        # torch中的view()的作用相当于numpy中的reshape，重新定义矩阵的形状。

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input1 = torch.matmul(h, self.a1)
        a_input2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(a_input1 + a_input2.transpose(-1, -2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        # nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。你可以把任意 nn.Module 的子类
        # (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，方法和 Python 自带的 list 一样，无非是 extend，append 等操作。但不同于
        # 一般的 list，加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中。若使用python的list，则会出问题。
        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(nhid, nclass, batch_norm=False))

    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.state_dim = state_dim

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, x, node_anchor_adj):
        a_in = torch.matmul(node_anchor_adj, x)
        a_out = torch.matmul(node_anchor_adj.transpose(1, 2), x)
        a = torch.cat((a_in, a_out, x), dim=2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * x), dim=2)
        h_hat = self.transform(joined_input)

        output = (1 - z) * x + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, n_edge_types, n_steps, nclass):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps
        self.nclass = nclass

        self.propagator = Propogator(self.state_dim)


        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, node_anchor_adj):
        for i_step in range(self.n_steps):
            x = self.propagator(x, node_anchor_adj)

        output = x

        return output

