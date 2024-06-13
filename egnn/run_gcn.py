
import torch
import torch.nn.functional as F
import argparse
from torch import nn
from dataloader.qm9 import dataQM9

import os
import torch
import torch.nn.functional as F
import time
import json

torch.manual_seed(233)

parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, default="torch")
parser.add_argument('--load-checkpoints', type=int, default=-1)
parser.add_argument('--save-interval', type=int, default=10)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--target', type=int, default=0)
args = parser.parse_args()

target = args.target
dim = 64
dataset = dataQM9(target=target)

print("Using torch backend")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models import GCN_torch
model = GCN_torch.GCN(dataset, dim).to(device)
loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)

def train():
    
    model.train()
    loss_all = 0

    for data in dataset.train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), data.y)
        loss.backward()
        loss_all += loss * data.num_graphs
        optimizer.step()
        
    return loss_all / len(dataset.train_loader.dataset) 

def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * dataset.std - data.y * dataset.std).abs().sum().item()  # MAE
    return error / len(loader.dataset)

if __name__ == '__main__':
    
    val_mae_list = []
    test_mae_list = []
    avg_epoch_time_list = []
    
    best_val_error = None
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    start_epoch = 1
    
    if args.load_checkpoints > 0:
        model.load_state_dict(torch.load(f'checkpoints/GCN_{args.backend}_{target}_{args.load_checkpoints}.pt'))
        start_epoch = args.load_checkpoints + 1
    
    total_time = 0
    for epoch in range(start_epoch, args.epoch + 1):
        
        start_time = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error = test(dataset.val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(dataset.test_loader)
            best_val_error = val_error

        epoch_time = time.time() - start_time
        total_time += epoch_time
        avg_epoch_time = total_time / epoch
        
        print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
              f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}, '
              f'Epoch Time: {time.time() - start_time:.7f}, Avg Epoch Time: {total_time / epoch:.7f}')
        
        val_mae_list.append(val_error)
        test_mae_list.append(test_error)
        avg_epoch_time_list.append(avg_epoch_time)
        
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f'checkpoints/GCN_{args.backend}_{args.target}_{epoch}.pt')
        
        with open(f'logs/GCN_{args.backend}_{args.target}.json', 'w') as f:
            json.dump({'val_mae': val_mae_list, 'test_mae': test_mae_list, 'avg_epoch_time': avg_epoch_time_list}, f)