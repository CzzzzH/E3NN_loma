import torch
import torch.nn as nn
import loma

import torch

loma_linear_ = loma.Linear()

class LomaLinearFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight, bias):
        outputs = loma_linear_.forward(input, weight, bias)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        outputs = loma_linear_.backward(grad_output.contiguous())
        d_input, d_weight, d_bias = outputs
        return d_input, d_weight, d_bias

class LomaLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(LomaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))

    def forward(self, input):
        return LomaLinearFunction.apply(input, self.weight, self.bias)
    
if __name__ == '__main__':

    bs = 4
    in_features = 10
    out_features = 10

    # model = torch.nn.Linear(in_fesatures, out_features)
    model = LomaLinear(in_features, out_features)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    model.train()

    for i in range(100):

        x = torch.randn(bs, in_features)
        label = x * 3 - 7
        pred = model(x)
        loss = loss_func(label, pred)
        print(f"Loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()