import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

class GraphLoss(nn.Module):
    def __init__(self, mu=0.1):
        super(GraphLoss, self).__init__()
        self.mu = mu

    def forward(self, output, target, data):
        if data.one_index:
            target = target - 1

        # Supervised node classification loss
        if data.train_mask:
            supervised_loss = F.nll_loss(output[data.train_mask], target[data.train_mask])
        else:
            supervised_loss = F.nll_loss(output, target)

        # Label smoothness regularization
        edge_index = data.edge_index
        node_preds_log = torch.log_softmax(output, dim=1)
        row, col = edge_index
        diff = output[row] - output[col]
        smoothness_loss = diff.pow(2).mean()

        loss = supervised_loss + self.mu * smoothness_loss

        return loss