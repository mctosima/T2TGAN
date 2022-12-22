import torch
import torch.nn as nn

# create ABP sequence decoder
class ABPDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(ABPDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h_n, c_n):
        # forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        # decode the hidden state of the last time step
        out = self.fc(out)
        return out, (h_n, c_n)