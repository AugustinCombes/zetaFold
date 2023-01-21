import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class GTN(nn.Module):
    """
    Encoder-transformer message passing layer
    """
    def __init__(self, input_dim, hidden_dim, dropout, n_class, device="cuda"):
        super(GTN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=2*hidden_dim, batch_first=True, dropout=0.3)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.device = device

    def forward(self, x_in, adj, idx, cum_nodes):
        # First message passing layer as embedding before the transformer encoder message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        # x = self.dropout(x)

        # Transformer encoder layer
        batch_lists = [[] for _ in range(len(cum_nodes))]
        for i, ind in enumerate(idx):
            batch_lists[ind].append(x[i])

        padded_batch = pad_sequence([torch.stack(b) for b in batch_lists], batch_first=True).to(x_in.device)
        pad_mask = torch.nn.utils.rnn.pad_sequence([torch.zeros(e) for e in cum_nodes], batch_first=True, padding_value=1).type(torch.bool).to(x_in.device)
        
        x = self.fc2(padded_batch, src_key_padding_mask=pad_mask)
        x = torch.vstack([x[j,:c] for j,c in enumerate(cum_nodes)])
        
        # Sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)
        
        # batch normalization layer
        # out = self.bn(out)

        # mlp to produce output
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        
        return F.log_softmax(out, dim=1)
