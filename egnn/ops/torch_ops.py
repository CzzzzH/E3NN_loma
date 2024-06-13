import torch
import math

from torch.nn import init
from . import *

class LomaAddFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input1, input2):
        outputs, input_ctx = loma_add_.forward(input1, input2)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_add_.backward(grad_output.contiguous(), *ctx.input_ctx)

class LomaAddBroadcastFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input1, input2):
        outputs, input_ctx = loma_add_broadcast_.forward(input1, input2)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_add_broadcast_.backward(grad_output.contiguous(), *ctx.input_ctx)

class LomaSubFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input1, input2):
        outputs, input_ctx = loma_sub_.forward(input1, input2)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_sub_.backward(grad_output.contiguous(), *ctx.input_ctx)
    
class LomaMulFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input1, input2):
        outputs, input_ctx = loma_mul_.forward(input1, input2)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_mul_.backward(grad_output.contiguous(), *ctx.input_ctx)

class LomaMulBroadcastFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input1, input2):
        outputs, input_ctx = loma_mul_broadcast_.forward(input1, input2)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_mul_broadcast_.backward(grad_output.contiguous(), *ctx.input_ctx)

class LomaDivFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input1, input2):
        outputs, input_ctx = loma_div_.forward(input1, input2)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_div_.backward(grad_output.contiguous(), *ctx.input_ctx)

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
    
class LomaMAELossFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, y):
        outputs, input_ctx = loma_mae_.forward(x, y)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        outputs = loma_mae_.backward(grad_output.contiguous(), *ctx.input_ctx)
        d_x, d_y = outputs
        return d_x, d_y
    
class LomaSumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        outputs, input_ctx = loma_sum_.forward(input)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_sum_.backward(grad_output.contiguous(), *ctx.input_ctx)

class LomaSumAggrFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, index):
        outputs, input_ctx = loma_sum_aggr_.forward(input, index)
        ctx.input_ctx = input_ctx
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return loma_sum_aggr_.backward(grad_output.contiguous(), *ctx.input_ctx)

class LomaAdd(torch.nn.Module):
    
    def __init__(self):
        super(LomaAdd, self).__init__()
        
    def forward(self, input1, input2):
        return LomaAddFunction.apply(input1, input2)

class LomaAddBroadcast(torch.nn.Module):
    
    def __init__(self):
        super(LomaAddBroadcast, self).__init__()
        
    def forward(self, input1, input2):
        return LomaAddBroadcastFunction.apply(input1, input2)

class LomaSub(torch.nn.Module):
    
    def __init__(self):
        super(LomaSub, self).__init__()
        
    def forward(self, input1, input2):
        return LomaSubFunction.apply(input1, input2)
    
class LomaMul(torch.nn.Module):
    
    def __init__(self):
        super(LomaMul, self).__init__()
        
    def forward(self, input1, input2):
        return LomaMulFunction.apply(input1, input2)

class LomaMulBroadcast(torch.nn.Module):
    
    def __init__(self):
        super(LomaMulBroadcast, self).__init__()
        
    def forward(self, input1, input2):
        return LomaMulBroadcastFunction.apply(input1, input2)

class LomaDiv(torch.nn.Module):
    
    def __init__(self):
        super(LomaDiv, self).__init__()
        
    def forward(self, input1, input2):
        return LomaDivFunction.apply(input1, input2)

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
    
class LomaMAELoss(torch.nn.Module):
    
    def __init__(self):
        super(LomaMAELoss, self).__init__()

    def forward(self, x, y):
        return LomaMAELossFunction.apply(x, y)
    
class LomaSum(torch.nn.Module):
    
    def __init__(self):
        super(LomaSum, self).__init__()

    def forward(self, input):
        return LomaSumFunction.apply(input)

class LomaSumAggr(torch.nn.Module):
    
    def __init__(self):
        super(LomaSumAggr, self).__init__()

    def forward(self, input, index):
        return LomaSumAggrFunction.apply(input, index)