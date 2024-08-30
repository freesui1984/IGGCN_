"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


class DiffusionGraphConv(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, bias_start=0.0,
                 filter_type='laplacian'):
        """
        Diffusion graph convolution
        Args:
            num_supports: number of supports, 1 for 'laplacian' filter and 2
                for 'dual_random_walk'
            input_dim: input feature dim
            hid_dim: hidden units
            num_nodes: number of nodes in graph
            max_diffusion_step: maximum diffusion step
            output_dim: output feature dim
            filter_type: 'laplacian' for undirected graph, and 'dual_random_walk'
                for directed graph
        """
        super(DiffusionGraphConv, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(
            torch.FloatTensor(
                size=(
                    self._input_size *
                    num_matrices,
                    output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, supports, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes,
        # input_dim/hidden_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # (batch, num_nodes, input_dim+hidden_dim)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size

        x0 = inputs_and_state  # (batch, num_nodes, input_dim+hidden_dim)
        # (batch, 1, num_nodes, input_dim+hidden_dim)
        x = torch.unsqueeze(x0, dim=1)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                # (batch, num_nodes, input_dim+hidden_dim)
                x1 = torch.matmul(support, x0)
                # (batch, ?, num_nodes, input_dim+hidden_dim)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    # (batch, num_nodes, input_dim+hidden_dim)
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(
                        x, x2)  # (batch, ?, num_nodes, input_dim+hidden_dim)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * \
            self._max_diffusion_step + 1  # Adds for x itself
        # (batch, num_nodes, num_matrices, input_hidden_size)
        x = torch.transpose(x, dim0=1, dim1=2)
        # (batch, num_nodes, input_hidden_size, num_matrices)
        x = torch.transpose(x, dim0=2, dim1=3)
        x = torch.reshape(
            x,
            shape=[
                batch_size,
                self._num_nodes,
                input_size *
                num_matrices])
        x = torch.reshape(
            x,
            shape=[
                batch_size *
                self._num_nodes,
                input_size *
                num_matrices])
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
            self,
            input_dim,
            num_units,
            max_diffusion_step,
            num_nodes,
            filter_type="laplacian",
            nonlinearity='tanh',
            use_gc_for_ru=True):
        """
        Args:
            input_dim: input feature dim
            num_units: number of DCGRU hidden units
            max_diffusion_step: maximum diffusion step
            num_nodes: number of nodes in the graph
            filter_type: 'laplacian' for undirected graph, 'dual_random_walk' for directed graph
            nonlinearity: 'tanh' or 'relu'. Default is 'tanh'
            use_gc_for_ru: decide whether to use graph convolution inside rnn. Default True
        """
        super(DCGRUCell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        if filter_type == "laplacian":  # ChebNet graph conv
            self._num_supports = 1
        elif filter_type == "random_walk":  # Forward random walk
            self._num_supports = 1
        elif filter_type == "dual_random_walk":  # Bidirectional random walk
            self._num_supports = 2
        else:
            self._num_supports = 1

        self.dconv_gate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2,
            filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def forward(self, supports, inputs, state):
        """
        Args:
            inputs: (B, num_nodes * input_dim)
            state: (B, num_nodes * num_units)
        Returns:
            output: (B, num_nodes * output_dim)
            state: (B, num_nodes * num_units)
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(supports, inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            value, split_size_or_sections=int(
                output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        # batch_size, self._num_nodes * output_size
        c = self.dconv_candidate(supports, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


class Propogator(nn.Module):
    """
    Gated Propagator for GGNN
    Using GRU gating mechanism similar to the previous implementation.
    """

    def __init__(self, state_dim):
        super(Propogator, self).__init__()
        self.state_dim = state_dim

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )

    def forward(self, x, supports):
        # Aggregate messages from neighbors
        a_in = torch.matmul(supports, x)
        a_out = torch.matmul(supports.transpose(1, 2), x)

        a_in = torch.stack(a_in, dim=-1).sum(dim=-1)
        a_out = torch.stack(a_out, dim=-1).sum(dim=-1)

        a = torch.cat((a_in, a_out, x), dim=2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * x), dim=2)
        h_hat = self.transform(joined_input)

        output = (1 - z) * x + z * h_hat

        return output


class GGNNLayer(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN) Layer
    """

    def __init__(self, input_dim, hidden_dim, num_nodes, num_steps, output_dim):
        super(GGNNLayer, self).__init__()
        self.num_steps = num_steps
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.propagator = Propogator(hidden_dim + input_dim)
        self.fc = nn.Linear(hidden_dim + input_dim, output_dim)

    def forward(self, supports, inputs, state):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))
        state = torch.reshape(state, (batch_size, self.num_nodes, self.hidden_dim))

        # Concatenate inputs and state
        x = torch.cat([inputs, state], dim=2)  # (batch_size, num_nodes, input_dim + hidden_dim)

        for _ in range(self.num_steps):
            x = self.propagator(x, supports)

        x = self.fc(x)  # (batch_size, num_nodes, output_dim)

        x = torch.reshape(x, [batch_size, self.num_nodes * self.output_dim])

        return x

class GCGRUCell(nn.Module):
    """
    Graph Convolutional Gated Recurrent Unit Cell using GGNN.
    """

    def __init__(
            self,
            input_dim,
            num_units,
            num_nodes,
            num_steps,
            nonlinearity='tanh'):
        """
        Args:
            input_dim: input feature dimension
            num_units: number of GCGRU hidden units
            num_nodes: number of nodes in the graph
            num_steps: number of GGNN steps
            nonlinearity: 'tanh' or 'relu'. Default is 'tanh'
        """
        super(GCGRUCell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units

        self.ggnn_gate = GGNNLayer(
            input_dim=input_dim,
            hidden_dim=num_units,
            num_nodes=num_nodes,
            num_steps=num_steps,
            output_dim=num_units * 2)
        self.ggnn_candidate = GGNNLayer(
            input_dim=input_dim,
            hidden_dim=num_units,
            num_nodes=num_nodes,
            num_steps=num_steps,
            output_dim=num_units)

    @property
    def output_size(self):
        return self._num_nodes * self._num_units

    def forward(self, supports, inputs, state):
        """
        Args:
            inputs: (B, num_nodes * input_dim)
            state: (B, num_nodes * num_units)
        Returns:
            output: (B, num_nodes * output_dim)
            state: (B, num_nodes * num_units)
        """
        output_size = 2 * self._num_units

        value = torch.sigmoid(
            self.ggnn_gate(supports, inputs, state))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, split_size_or_sections=int(output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self.ggnn_candidate(supports, inputs, r * state)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c

        return output, new_state

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)

