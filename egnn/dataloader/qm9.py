import copy
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops


class Complete:
    def __call__(self, data):
        data = copy.copy(data)
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class dataQM9:
    def __init__(self, target=0):
        self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
        self.target = target
        class MyTransform:
            def __call__(self, data):
                data = copy.copy(data)
                data.y = data.y[:, target]  # Specify target.
                return data
        self.transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
        self.get_data()

    def get_data(self):
        dataset = QM9(self.path, transform=self.transform).shuffle()

        # Normalize targets to mean = 0 and std = 1.
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        mean, std = mean[:, target].item(), std[:, target].item()

        # Split datasets.
        test_dataset = dataset[:10000]
        val_dataset = dataset[10000:20000]
        train_dataset = dataset[20000:]
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    
# data.x, input node features, shape = [num_nodes, num_node_features]
# data.edge_index, graph connectivity in COO format with shape [2, num_edges]
# data.edge_attr, edge features, shape = [num_edges, num_edge_features]

if __name__ == "__main__":
    target = 0
    data = dataQM9(target)
    for batch in data.train_loader:
        print(batch)
        break