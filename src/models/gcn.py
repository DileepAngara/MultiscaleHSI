import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

class GCN(nn.Module):
  def __init__(self, nfeat, nhid, nout, n_nodes, dropout):
    super(GCN, self).__init__()
    self.conv1 = GCNConv(nfeat, nhid)
    self.dropout = nn.Dropout(dropout)
    self.conv2 = GCNConv(nhid, nhid)
    self.conv3 = GCNConv(nhid, nout)

  def forward(self, data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

    x = self.conv1(x, edge_index, edge_attr)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.conv2(x, edge_index, edge_attr)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.conv3(x, edge_index, edge_attr)
    x = F.log_softmax(x, dim=1)
    return x