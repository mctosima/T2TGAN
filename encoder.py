import torch
import torch.nn as nn

"""
Remember, for each subject, we have 3 column signal data.
First is PPG, Second is ABP, Third is ECG.
We want to make a seq2seq model that can predict ABP from PPG and ECG.
"""

# create ABP sequence encoder
class ABPEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(ABPEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        # return the final hidden state and cell state
        return h_n, c_n

