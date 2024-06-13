import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, aggr

class GCN(nn.Module):
    def __init__(self, dataset, dim):
        super(GCN, self).__init__()
        self.lin0 = nn.Linear(dataset.num_features, dim)
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.conv3 = GCNConv(dim, dim)
        self.sum_aggr = aggr.SumAggregation()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, 1)

    def forward(self, data):
        x = F.relu(self.lin0(data.x))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))
        x = self.sum_aggr(x, data.batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.view(-1)