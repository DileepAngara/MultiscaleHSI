import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, dropout):
        super(MPNN_LSTM, self).__init__()
        self.nhid = nhid
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.lstm1 = nn.LSTM(nhid * 2 + nfeat, nhid, 1, bidirectional=True)
        self.lstm2 = nn.LSTM(nhid * 2, nhid, 1, bidirectional=True)
        self.fc1 = nn.Linear(nhid * 4 + nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, adj, weight = data.x, data.edge_index, data.edge_attr
        skip = x  # Skip connection
        lst = [x]

        x = F.relu(self.conv1(x, adj, weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = F.relu(self.conv2(x, adj, weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        x = torch.cat(lst, dim=1)
        x = x.unsqueeze(0)  # Add sequence dimension

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)  # Pass only out1 to lstm2

        # Concatenate forward and backward outputs
        x = torch.cat([out1[0,:,:],out2[0,:,:]], dim=1)

        x = torch.cat([x, skip], dim=1)  # Concatenate along the node dimension

        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x