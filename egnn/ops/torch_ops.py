import torch
from . import *

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