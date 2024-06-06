import copy
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set, aggr
from torch_geometric.utils import remove_self_loops
from dataloader.qm9 import dataQM9



import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Net(nn.Module):
    def __init__(self, dataset, dim):
        super(Net, self).__init__()
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

target = 0
dim = 64
dataset = dataQM9(target=target)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset, dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in dataset.train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(dataset.train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * dataset.std - data.y * dataset.std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(dataset.val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(dataset.test_loader)
        best_val_error = val_error

    print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
          f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')