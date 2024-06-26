import torch
import argparse
import time

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(233)

parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, default="loma")
args = parser.parse_args()

if args.backend == "loma":
    print("Using loma backend")
    device = 'cpu'
    from models import MLP_loma
    from ops.torch_ops import LomaMSELoss
    model = MLP_loma.MLP()
    loss_fn = LomaMSELoss()
else:
    print("Using torch backend")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    from models import MLP_torch
    model = MLP_torch.MLP().to(device)
    loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader):

    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        pred = pred.reshape(-1, 1)
        y = y.reshape(-1, 1).float()
        loss = loss_fn(pred, y)
    
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            pred = pred.reshape(-1, 1)
            y = y.reshape(-1, 1).float()
            test_loss += loss_fn(pred, y).item()
            correct += ((pred - y) < 0.5).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    epochs = 1
    begin_time = time.time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader)
        test(test_dataloader)
    print(f"Done! Total Time: {time.time() - begin_time}s")