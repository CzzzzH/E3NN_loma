import os 
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(os.path.join(parent, 'loma_public'))
import compiler

from ctypes import CDLL
from .utils import *

class Module:
    
    def __init__(self, *args):
        
        func_name, lib_name, target = args
        self.func_name = func_name

        if target == 'ispc':
            if os.path.exists(f'{current}/_code/{lib_name}.so'):
                self.lib = CDLL(f'{current}/_code/{lib_name}.so')
            else:
                with open(f'{current}/loma_code/{lib_name}.py') as f:
                    _, self.lib = compiler.compile(f.read(),
                                            target = 'ispc',
                                            output_filename = f'{current}/_code/{lib_name}')
        elif target == 'opencl':
            # TODO: Implement OpenCL compilation
            raise NotImplementedError('OpenCL compilation is not implemented yet')
        else:
            raise ValueError('target should be either "ispc" or "opencl"')

    def save_input(self, *input):
        self.input_ctx = input

    def forward(self, *input):
        raise NotImplementedError('forward method is not implemented')

    def backward(self, *grad_output):
        raise NotImplementedError('backward method is not implemented')

    def __call__(self, *input):
        return self.forward(*input)

class UnaryModule(Module):

    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, input):

        bs, num_features = input.shape
        num_threads = bs * num_features
        input_ctype = build_ctypes(input, bs * num_features)
        output_ctype = (ctypes.c_float * (bs * num_features))(0)
        getattr(self.lib, self.func_name)(input_ctype, output_ctype, num_features, num_threads)
        output = build_tensor(output_ctype, (bs, num_features))
        self.save_input(input_ctype, bs, num_features, num_threads)
        return output

    def backward(self, grad_output):

        input_ctype, bs, num_features, num_threads = self.input_ctx
        output_ctype = build_ctypes(grad_output, bs * num_features)
        grad_input_ctype = (ctypes.c_float * (bs * num_features))(0)
        getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_ctype, output_ctype, num_features, ctypes.c_int(0), num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, num_features))
        return grad_input

class ReLU(UnaryModule):

    def __init__(self, lib_name='ops', target='ispc'):\
        super().__init__('relu', lib_name, target)
    
    def forward(self, input):
        return super().forward(input)

    def backward(self, grad_output):
        return super().backward(grad_output)

class SiLU(UnaryModule):

    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('silu', lib_name, target)
    
    def forward(self, input):
        return super().forward(input)

    def backward(self, grad_output):
        return super().backward(grad_output)

class Linear(Module):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('linear', lib_name, target)
    
    def forward(self, input, weight, bias):

        bs, in_features = input.shape
        out_features = weight.shape[0]
        num_threads = bs * out_features
        input_ctype = build_ctypes(input, bs * in_features)
        weight_ctype = build_ctypes(weight, out_features * in_features)
        bias_ctype = build_ctypes(bias, out_features)
        output_ctype = (ctypes.c_float * (bs * out_features))(0)
        getattr(self.lib, self.func_name)(input_ctype, weight_ctype, bias_ctype, output_ctype, in_features, out_features, num_threads)
        output = build_tensor(output_ctype, (bs, out_features))
        self.save_input(input_ctype, weight_ctype, bias_ctype, bs, in_features, out_features, num_threads)
        return output

    def backward(self, grad_output):

        input_ctype, weight_ctype, bias_ctype, bs, in_features, out_features, num_threads = self.input_ctx
        grad_input_ctype = (ctypes.c_float * (bs * in_features))(0)
        grad_weight_ctype = (ctypes.c_float * (out_features * in_features))(0)
        grad_bias_ctype = (ctypes.c_float * (out_features))(0)
        output_ctype = build_ctypes(grad_output, bs * out_features)
        getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_ctype, weight_ctype, grad_weight_ctype, 
                bias_ctype, grad_bias_ctype, output_ctype, in_features, ctypes.c_int(0), out_features, ctypes.c_int(0), 
                num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, in_features))
        return grad_input