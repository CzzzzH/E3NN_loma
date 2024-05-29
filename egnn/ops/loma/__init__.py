import os 
import sys

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current, 'loma_public'))
import compiler

from ctypes import CDLL
from .utils import *

def get_shape(reduce_level, *args):
    return [args[i] for i in range(reduce_level)]

def get_length(reduce_level, *args):
    length = 1
    for i in range(reduce_level):
        length *= args[i]
    return length

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
        elif target == 'c':
            if os.path.exists(f'{current}/_code/{lib_name}.so'):
                self.lib = CDLL(f'{current}/_code/{lib_name}.so')
            else:
                with open(f'{current}/loma_code/{lib_name}.py') as f:
                    _, self.lib = compiler.compile(f.read(),
                                            target = 'c',
                                            output_filename = f'{current}/_code/{lib_name}')
        elif target == 'opencl':
            # TODO: Implement OpenCL compilation
            raise NotImplementedError('OpenCL compilation is not implemented yet')
        else:
            raise ValueError('target should be either "c" or "ispc" or "opencl"')

    def forward(self, *args):
        raise NotImplementedError('forward method is not implemented')

    def backward(self, *args):
        raise NotImplementedError('backward method is not implemented')

    def __call__(self, *args):
        return self.forward(*args)

class UnaryModule(Module):

    def __init__(self, *args):
        super().__init__(*args)
        func_name, _, _ = args
        if func_name in ['sum_', 'mean_']:
            self.reduce = 1
        else:
            self.reduce = 2
        
    def forward(self, input):

        bs, num_features = input.shape
        num_threads = bs * num_features
        input_ctype = build_ctypes(input, bs * num_features) 
        output_ctype = (ctypes.c_float * get_length(self.reduce, bs, num_features))(0)
        getattr(self.lib, self.func_name)(input_ctype, output_ctype, num_features, num_threads)
        output = build_tensor(output_ctype, get_shape(self.reduce, bs, num_features))
        return output, (input_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        input_ctype, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, get_length(self.reduce, bs, num_features))
        grad_input_ctype = (ctypes.c_float * (bs * num_features))(0)
        getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_ctype, output_ctype, num_features, ctypes.c_int(0), num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, num_features))
        return grad_input

class Sqrt(UnaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('sqrt_', lib_name, target)

class Sum(UnaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('sum_', lib_name, target)

class Mean(UnaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('mean_', lib_name, target)

class ReLU(UnaryModule):

    def __init__(self, lib_name='ops', target='ispc'):\
        super().__init__('relu_', lib_name, target)
    
class SiLU(UnaryModule):

    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('silu_', lib_name, target)

class Sigmoid(UnaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('sigmoid_', lib_name, target)

class BinaryModule(Module):

    def __init__(self, *args):
        super().__init__(*args)
        func_name, _, _ = args
        self.reduce = 2
        
    def forward(self, x, y):

        bs, num_features = x.shape
        num_threads = bs * num_features
        x_ctype = build_ctypes(x, bs * num_features) 
        y_ctype = build_ctypes(y, bs * num_features)
        output_ctype = (ctypes.c_float * get_length(self.reduce, bs, num_features))(0)
        getattr(self.lib, self.func_name)(x_ctype, y_ctype, output_ctype, num_features, num_threads)
        output = build_tensor(output_ctype, get_shape(self.reduce, bs, num_features))
        return output, (x_ctype, y_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        x_ctype, y_ctype, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, get_length(self.reduce, bs, num_features))
        grad_x_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_y_ctype = (ctypes.c_float * (bs * num_features))(0)
        getattr(self.lib, f'grad_{self.func_name}')(x_ctype, grad_x_ctype, y_ctype, grad_y_ctype, output_ctype, num_features, ctypes.c_int(0), num_threads)
        grad_x = build_tensor(grad_x_ctype, (bs, num_features))
        grad_y = build_tensor(grad_y_ctype, (bs, num_features))
        return grad_x, grad_y
    
class Add(BinaryModule):

    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('add_', lib_name, target)

class Sub(BinaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('sub_', lib_name, target)
        
class Mul(BinaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('multiply_', lib_name, target)

class Div(BinaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('divide_', lib_name, target)

class MSELoss(BinaryModule):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('mse_', lib_name, target)
        
    def forward(self, x, y):

        bs, num_features = x.shape
        num_threads = bs * num_features
        x_ctype = build_ctypes(x, bs * num_features) 
        y_ctype = build_ctypes(y, bs * num_features)
        output_ctype = (ctypes.c_float * 1)(0)
        getattr(self.lib, self.func_name)(x_ctype, y_ctype, output_ctype, bs, num_features, num_threads)
        output = build_tensor(output_ctype, [])
        return output, (x_ctype, y_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        x_ctype, y_ctype, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, 1)
        grad_x_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_y_ctype = (ctypes.c_float * (bs * num_features))(0)
        getattr(self.lib, f'grad_{self.func_name}')(x_ctype, grad_x_ctype, y_ctype, grad_y_ctype, output_ctype, bs, ctypes.c_int(0), 
                                                        num_features, ctypes.c_int(0), num_threads)
        grad_x = build_tensor(grad_x_ctype, (bs, num_features))
        grad_y = build_tensor(grad_y_ctype, (bs, num_features))
        return grad_x, grad_y

class Linear(Module):
    
    def __init__(self, lib_name='ops', target='ispc'):
        super().__init__('linear_', lib_name, target)
    
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
        return output, (input_ctype, weight_ctype, bias_ctype, bs, in_features, out_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        input_ctype, weight_ctype, bias_ctype, bs, in_features, out_features, num_threads = input_ctx
        grad_input_ctype = (ctypes.c_float * (bs * in_features))(0)
        grad_weight_ctype = (ctypes.c_float * (out_features * in_features))(0)
        grad_bias_ctype = (ctypes.c_float * (out_features))(0)
        output_ctype = build_ctypes(grad_output, bs * out_features)
        getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_ctype, weight_ctype, grad_weight_ctype, 
                bias_ctype, grad_bias_ctype, output_ctype, in_features, ctypes.c_int(0), out_features, ctypes.c_int(0), 
                num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, in_features))
        grad_weight = build_tensor(grad_weight_ctype, (out_features, in_features))
        grad_bias = build_tensor(grad_bias_ctype, (out_features,))
        return grad_input, grad_weight, grad_bias