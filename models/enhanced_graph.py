import torch.nn as nn
import torch
import torch.nn.functional as F

class GTN(nn.Module):
    """
    Encoder-transformer message passing layer
    """
    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GTN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=256, batch_first=True)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx, cum_nodes):
        # embedding layer
        x = self.embedding(x_in)

        # first message passing layer
        pad_mask = torch.nn.utils.rnn.pad_sequence([torch.zeros(e) for e in cum_nodes], batch_first=True, padding_value=1).type(torch.bool)
        x = self.fc1(x, src_key_padding_mask=pad_mask)
        x = torch.vstack([x[j,:c] for j,c in enumerate(cum_nodes)])
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer #keep 
        # x = self.fc2(x)
        # x = self.relu(torch.mm(adj, x))
        
        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)
        
        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        
        return F.log_softmax(out, dim=1)
