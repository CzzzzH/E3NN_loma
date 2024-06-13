import torch

from torch import nn
from ops.torch_ops import LomaLinear, LomaReLU, LomaSub, LomaAddBroadcast, LomaMulBroadcast, LomaSum, LomaSumAggr

class EGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(EGNNLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            # edge_attr is 5-dimensional
            LomaLinear(2 * in_features + 5, out_features),
            LomaReLU(),
            LomaLinear(out_features, out_features)
        )
        self.node_mlp = nn.Sequential(
            LomaLinear(in_features + out_features, out_features),
            LomaReLU(),
            LomaLinear(out_features, out_features)
        )
        self.coord_mlp = nn.Sequential(
            LomaLinear(out_features, 1),
            LomaReLU(),
            LomaLinear(1, 1)
        )
        
        self.sum = LomaSum()
        self.sum_aggr = LomaSumAggr()
        self.sub = LomaSub()
        self.add_broadcast = LomaAddBroadcast()
        self.mul_broadcast = LomaMulBroadcast()

    def forward(self, x, edge_index, edge_attr, pos):
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_features = self.edge_mlp(edge_input)
        
        aggregated_edge_features = self.sum_aggr(edge_features, row)
        node_input = torch.cat([x, aggregated_edge_features], dim=1)
        node_features = self.node_mlp(node_input)
        
        relative_pos = self.sub(pos[row], pos[col])
        coord_updates = self.mul_broadcast(self.coord_mlp(edge_features), relative_pos)
        coord_updates = self.sum(coord_updates)
        pos = self.add_broadcast(pos, coord_updates)

        return node_features, pos

class EGNN(nn.Module):
    def __init__(self, dataset, dim):
        super(EGNN, self).__init__()
        self.lin0 = LomaLinear(dataset.num_features, dim)
        self.egnn1 = EGNNLayer(dim, dim)
        self.egnn2 = EGNNLayer(dim, dim)
        self.egnn3 = EGNNLayer(dim, dim)
        self.sum_aggr = LomaSumAggr()
        self.lin1 = LomaLinear(dim, dim)
        self.lin2 = LomaLinear(dim, 1)
        self.relu = LomaReLU()

    def forward(self, data):
        x = self.relu(self.lin0(data.x))
        pos = data.pos

        x, pos = self.egnn1(x, data.edge_index, data.edge_attr, pos)
        x, pos = self.egnn2(x, data.edge_index, data.edge_attr, pos)
        x, pos = self.egnn3(x, data.edge_index, data.edge_attr, pos)

        x = self.sum_aggr(x, data.batch)
        x = self.relu(self.lin1(x))
        x = self.lin2(x)
        return x.view(-1)
    
    def get_cl_mem(self):
        return self.relu.cl_mem