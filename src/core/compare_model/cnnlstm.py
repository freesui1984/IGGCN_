import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=2, device=None):
        super(CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        self.device = device

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fc1 = nn.Linear(1536, 512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2)
        self.fc2 = nn.Linear(128, num_classes)

    def last_relevant_pytorch(self, output, lengths, batch_first=True):
        lengths = lengths.cpu()

        # masks of the true seq lengths
        masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
        time_dimension = 1 if batch_first else 0
        masks = masks.unsqueeze(time_dimension)
        masks = masks.to(output.device)
        last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
        last_output.to(output.device)

        return last_output

    def forward(self, x, seq_lengths):
        # x: (batch_size, num_nodes, input_dim)
        # seq_lengths: (batch_size, )

        batch_size, num_nodes, input_dim = x.shape

        # Reshape x to (batch_size * num_nodes, 1, input_dim, 1)
        x = x.view(batch_size * num_nodes, 1, input_dim, 1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)

        # Reshape out to (batch_size * num_nodes, -1)
        out = out.view(batch_size * num_nodes, -1)
        out = self.fc1(out)

        # Reshape out to (batch_size, num_nodes, -1)
        out = out.view(batch_size, num_nodes, -1)

        # LSTM
        lstm_out, _ = self.lstm(out)

        # Compute the last relevant output for each sequence
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long).to(self.device)
        lstm_out_last = self.last_relevant_pytorch(lstm_out, seq_lengths_tensor, batch_first=True)

        logits = self.fc2(lstm_out_last)

        # logits = F.sigmoid(logits)

        return logits



