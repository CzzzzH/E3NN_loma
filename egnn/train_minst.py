import torch
import torch.nn as nn

from models.MLP import MLP
    
if __name__ == '__main__':

    bs = 4
    in_features = 10
    out_features = 10

    # model = torch.nn.Linear(in_fesatures, out_features)
    model = MLP()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    model.train()

    for i in range(100):

        x = torch.randn(bs, in_features)
        label = x * 3 + 7
        pred = model(x)
        loss = loss_func(label, pred)
        print(f"Loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()