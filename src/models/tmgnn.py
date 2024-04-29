import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from torch_geometric.utils import to_dense_adj
import mpnn_encoder as MPNN_Encoder

class TMGNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, dropout, num_clusters = [5, 10], use_norm = False):
        super(TMGNN, self).__init__()
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.nout = nout
        self.use_norm = use_norm

        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+

        # Bottom encoder
        self.bottom_encoder = MPNN_Encoder(nfeat, nhid, nhid, dropout)

        # Number of clusters
        self.num_clusters = num_clusters

        # Multiresolution construction
        self.middle_linear = nn.ModuleList()
        self.middle_encoder = nn.ModuleList()

        for size in self.num_clusters:
            self.middle_linear.append(nn.Linear(nhid, size))
            self.middle_encoder.append(nn.Linear(nhid, nhid))

        # Mixing multiple resolutions together
        self.mix_1 = nn.Linear((len(self.num_clusters) + 1) * nhid, 512)
        self.mix_2 = nn.Linear(512, (len(self.num_clusters) + 1) * nhid)

        # +------------------------+
        # | Long Short-Term Memory |
        # +------------------------+

        self.rnn1 = nn.LSTM((len(self.num_clusters) + 1) * nhid, nhid, 1, bidirectional=True)
        self.rnn2 = nn.LSTM(nhid * 2, nhid, 1, bidirectional=True)

        self.fc1 = nn.Linear(nhid * 4 + nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        # +--------------------------------------+
        # | Multiresolution Graph Networks (MGN) |
        # +--------------------------------------+
        x, adj, weight = data.x, data.edge_index, data.edge_attr
        skip = x

        # All latents
        all_latents = []

        # Bottom encoder
        bottom_latent = self.bottom_encoder(x, adj, weight)
        all_latents.append(bottom_latent)

        # Product of all assignment matrices
        product = None

        # Multiresolution construction
        adj = to_dense_adj(adj)[0]
        latent = bottom_latent

        for level, size in enumerate(self.num_clusters):

            # Assignment matrix
            assign = self.middle_linear[level](latent)
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
            adj = torch.matmul(assign.transpose(0, 1), adj)
            adj = torch.matmul(adj, assign)
            adj = adj / torch.sum(adj)
            # New latent by graph convolution
            latent = torch.tanh(self.middle_encoder[level](torch.matmul(adj, x)))

            # Extended latent
            extended_latent = torch.matmul(product, latent)
            all_latents.append(extended_latent)

        # Normalization
        if self.use_norm == True:
            for idx in range(len(all_latents)):
                all_latents[idx] = all_latents[idx] / torch.norm(all_latents[idx], p = 2)

        # Concatenate all resolutions
        representation = torch.cat(all_latents, dim = 1)
        x = representation

        # Mixing multiple resolutions
        x = torch.relu(self.mix_1(x))
        x = torch.relu(self.mix_2(x))

        # +------------------------+
        # | Long Short-Term Memory |
        # +------------------------+

        x = x.unsqueeze(0)
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()

        out1, _ = self.rnn1(x)
        out2, _ = self.rnn2(out1)  # Pass only out1 to lstm2

        # Concatenate forward and backward outputs
        x = torch.cat([out1[0,:,:],out2[0,:,:]], dim=1)

        x = torch.cat([x, skip], dim=1)  # Concatenate along the node dimension

        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x