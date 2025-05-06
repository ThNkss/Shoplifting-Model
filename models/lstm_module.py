import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModule(nn.Module):
    def __init__(self, hidden_dim, input_dim=1280, num_layers=2, dropout=0.3, bidirectional=True):  # Increased layers
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            #dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # Sort sequences by length (for better packing)
        lengths, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # Reverse sorting
        _, unperm_idx = perm_idx.sort(0)
        out = out[unperm_idx]
        
        return self.dropout(out)