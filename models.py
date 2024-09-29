import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.models import GCN
  
class MGNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, num_clusters = [10, 5], use_norm = False):
        super(MGNN, self).__init__()
        self.nhid = nhid
        self.nfeat = nfeat
        self.nout = nout
        self.use_norm = use_norm

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+

        # Bottom encoder
        self.bottom_encoder = GCN(nfeat, nhid, 2, nhid, norm="layernorm")

        # Number of clusters
        self.num_clusters = num_clusters

        # Multiresolution construction
        self.middle_encoders = nn.ModuleList()
        self.middle_pools = nn.ModuleList()

        for size in self.num_clusters:
            self.middle_encoders.append(nn.Linear(nhid, nhid))
            self.middle_pools.append(nn.Linear(nhid, size))

        # Mixing multiple resolutions together
        self.fc = nn.Linear((len(self.num_clusters) + 1) * nhid, nout)

        self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, adj, weight=None):
        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+

        assert len(self.num_clusters) > 0, "Need at least one cluster"

        # All latents
        all_latents = []

        # Bottom encoder
        latent = self.bottom_encoder(x, adj, weight)
        latent = F.relu(latent)
        all_latents.append(latent)

        adj = to_dense_adj(adj, edge_attr=weight)[0]

        # Product of all assignment matrices
        product = None

        for level, size in enumerate(self.num_clusters):
            # Assignment matrix
            assign = self.middle_pools[level](latent)
            assign = F.gumbel_softmax(assign, tau = 1, hard = True, dim = 1)

            # Update product
            if level == 0:
                product = assign
            else:
                product = torch.matmul(product, assign)

            # Coarsen node features
            x = torch.matmul(assign.transpose(0, 1), latent)
            x = F.normalize(x, dim = 1)

            # Coarsen the adjacency
            adj = torch.matmul(torch.matmul(assign.transpose(0, 1), adj), assign)
            adj = adj / torch.sum(adj + 1e-8)

            # New latent by graph convolution
            latent = F.relu(self.middle_encoders[level](torch.matmul(adj, x)))

            # Extended latent
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm:
            all_latents = [F.normalize(latent, p=2, dim=1) for latent in all_latents]

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim=1)
        x = representation

        # Mixing multiple resolutions
        x = self.fc(x)

        return x