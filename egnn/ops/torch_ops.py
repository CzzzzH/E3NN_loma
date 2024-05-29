import torch
import math

from torch.nn import init
from . import *

class LomaLinearFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight, bias):
        outputs, input_ctx = loma_linear_.forward(input, weight, bias)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        outputs = loma_linear_.backward(grad_output.contiguous(), *ctx.input_ctx)
        d_input, d_weight, d_bias = outputs
        return d_input, d_weight, d_bias

class LomaReLUFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        output, input_ctx = loma_relu_.forward(input)
        ctx.input_ctx = input_ctx
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return loma_relu_.backward(grad_output.contiguous(), *ctx.input_ctx)

class LomaMSELossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        outputs, input_ctx = loma_mse_.forward(x, y)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        outputs = loma_mse_.backward(grad_output.contiguous(), *ctx.input_ctx)
        d_x, d_y = outputs
        return d_x, d_y

class LomaLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(LomaLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return LomaLinearFunction.apply(input, self.weight, self.bias)

class LomaReLU(torch.nn.Module):

    def __init__(self):
        super(LomaReLU, self).__init__()

    def forward(self, input):
        return LomaReLUFunction.apply(input)

class LomaMSELoss(torch.nn.Module):

    def __init__(self):
        super(LomaMSELoss, self).__init__()

    def forward(self, x, y):
        return LomaMSELossFunction.apply(x, y)