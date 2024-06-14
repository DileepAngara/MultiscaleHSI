import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
  def __init__(self, nfeat, nhid, nout, n_nodes, dropout):
    super(GCN, self).__init__()
    self.conv1 = GCNConv(nfeat, nhid)
    self.conv2 = GCNConv(nhid, nout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

    x = self.conv1(x, edge_index, edge_attr)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.conv2(x, edge_index, edge_attr)

    return F.log_softmax(x, dim=1)