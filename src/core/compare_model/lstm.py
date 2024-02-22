import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, num_nodes, rnn_units, num_rnn_layers, input_dim, num_classes, dropout, device=None):
        super(LSTMModel, self).__init__()

        self._num_nodes = num_nodes
        self._rnn_units = rnn_units
        self._num_rnn_layers = num_rnn_layers
        self._input_dim = input_dim
        self._num_classes = num_classes
        self._dropout = dropout
        self._device = device

        self.lstm = nn.LSTM(input_dim, rnn_units, num_rnn_layers, batch_first=True)
        self.dropout = nn.Dropout(p=self._dropout)
        self.fc = nn.Linear(rnn_units, num_classes)
        self.relu = nn.ReLU()

    def last_relevant_pytorch(self, output, lengths, batch_first=True):
        lengths = lengths.cpu()

        # masks of the true seq lengths
        _, _, size = output.shape
        masks = (lengths - 1).view(-1, 1).expand(len(lengths), size)
        time_dimension = 1 if batch_first else 0
        masks = masks.unsqueeze(time_dimension)
        masks = masks.to(output.device)
        last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
        last_output.to(output.device)

        return last_output

    def forward(self, inputs, seq_lengths):
        # inputs: (batch_size, num_nodes, input_dim)
        # seq_lengths: (batch_size, )

        batch_size, num_nodes, input_dim = inputs.shape

        # Reshape inputs to (batch_size * num_nodes, input_dim)
        inputs = inputs.view(batch_size * num_nodes, input_dim)

        # Initialize hidden states
        initial_hidden_state, initial_cell_state = self.init_hidden(batch_size * num_nodes)

        # LSTM
        output, _ = self.lstm(inputs.unsqueeze(1), (initial_hidden_state, initial_cell_state))

        # Reshape output to (batch_size, num_nodes, rnn_units)
        output = output.view(batch_size, num_nodes, -1)

        # Compute the last relevant output for each sequence
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long).to(self._device)
        last_out = self.last_relevant_pytorch(output, seq_lengths_tensor)

        # Dropout -> ReLU -> FC
        logits = self.fc(self.relu(self.dropout(last_out)))

        return logits

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        cell = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        return hidden, cell
