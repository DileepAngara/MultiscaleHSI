from torch_geometric.utils import degree
import torch.nn as nn
import torch.nn.functional as F


class GraphLoss(nn.Module):
    def __init__(self, mu=0.01):
        super(GraphLoss, self).__init__()
        self.mu = mu

    def forward(self, output, target, data):

        # Supervised node classification loss
        if data.train_mask:
            supervised_loss = F.nll_loss(
                output[data.train_mask], target[data.train_mask]
            )
        else:
            supervised_loss = F.nll_loss(output, target)

        # Label smoothness regularization
        edge_index = data.edge_index
        row, col = edge_index

        # Normalize the output values
        node_degrees = degree(row, num_nodes=data.x.size(0))
        output = output / node_degrees.view(-1, 1).sqrt()

        diff = output[row] - output[col]
        smoothness_loss = diff.pow(2).mean()

        loss = supervised_loss + self.mu * smoothness_loss

        return loss
