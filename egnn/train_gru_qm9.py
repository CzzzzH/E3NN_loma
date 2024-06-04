import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set
from dataloader.qm9 import dataQM9
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(torch.nn.Module):
    
    def __init__(self, dataset, dim):
        super().__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

def train(model, dataset, optimizer):
    
    model.train()
    loss_all = 0
    loss_it = 0

    for i, data in enumerate(dataset.train_loader):
        print(data)
        data.stop()
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        loss_it += loss.item() * data.num_graphs
        optimizer.step()
        
    return loss_all / len(dataset.train_loader.dataset)

def test(model, dataloader, std):
    
    model.eval()
    error = 0
    for data in dataloader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    writer.add_scalar('Test MAE', error / len(dataloader.dataset))
    return error / len(dataloader.dataset)

if __name__ == '__main__':
    
    target = 0
    dim = 64
    
    dataset = dataQM9(target=target)
    model = Net(dataset=dataset, dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.7, patience=5,
                                                        min_lr=0.00001)
    
    best_val_error = None
    writer = SummaryWriter(log_dir=f'runs/gru_qm9')
    
    for epoch in range(1, 301):
        
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(model, dataset, optimizer)
        val_error = test(model, dataset.val_loader, dataset.std)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(model, dataset.test_loader, dataset.std)
            best_val_error = val_error
            
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Val MAE', val_error, epoch)
        print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')