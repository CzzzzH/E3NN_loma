import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import Set2Set

class EGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(EGNNLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            # edge_attr is 5-dimensional
            nn.Linear(2 * in_features + 5, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(out_features, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )

    def forward(self, x, edge_index, edge_attr, pos):
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_features = self.edge_mlp(edge_input)
        
        aggregated_edge_features = torch.zeros_like(x)
        aggregated_edge_features.index_add_(0, row, edge_features)
        
        node_input = torch.cat([x, aggregated_edge_features], dim=1)
        node_features = self.node_mlp(node_input)
        
        relative_pos = pos[row] - pos[col]
        coord_updates = self.coord_mlp(edge_features) * relative_pos
        coord_updates = coord_updates.sum(dim=0, keepdim=True)
        
        pos = pos + coord_updates
        
        return node_features, pos

class EGNN(nn.Module):
    def __init__(self, dataset, dim):
        super(EGNN, self).__init__()
        self.lin0 = nn.Linear(dataset.num_features, dim)
        self.egnn1 = EGNNLayer(dim, dim)
        self.egnn2 = EGNNLayer(dim, dim)
        self.egnn3 = EGNNLayer(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)

    def forward(self, data):
        x = F.relu(self.lin0(data.x))
        pos = data.pos

        x, pos = self.egnn1(x, data.edge_index, data.edge_attr, pos)
        x, pos = self.egnn2(x, data.edge_index, data.edge_attr, pos)
        x, pos = self.egnn3(x, data.edge_index, data.edge_attr, pos)

        x = self.set2set(x, data.batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.view(-1)