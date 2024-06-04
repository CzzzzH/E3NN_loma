import os 
import sys

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current, 'loma_public'))
import compiler
import cl_utils
import gpuctypes.opencl as cl

from ctypes import CDLL
from .utils import *


TARGET = 'ispc'

class Module:
    
    def __init__(self, *args):
        
        func_name, lib_name, target = args
        self.func_name = func_name
        self.target = target

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
            cl_ctx, cl_device, cl_cmd_queue = cl_utils.create_context()
            with open(f'{current}/loma_code/{lib_name}.py') as f:
                _, self.lib = compiler.compile(f.read(),
                                        target = 'opencl',
                                        opencl_context = cl_ctx,
                                        opencl_device = cl_device,
                                        opencl_command_queue = cl_cmd_queue)
            self.cl_ctx = cl_ctx
            self.cl_device = cl_device
            self.cl_cmd_queue = cl_cmd_queue
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
        func_name, _, target = args
        if func_name in ['sum_', 'mean_']:
            self.reduce = 1
        else:
            self.reduce = 2
        self.target = target
        
    def forward(self, input):

        bs, num_features = input.shape
        num_threads = bs * num_features
        input_ctype = build_ctypes(input, bs * num_features) 
        output_ctype = (ctypes.c_float * (bs * num_features))(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(input_ctype), ctypes.byref(input_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            getattr(self.lib, self.func_name)(input_cl, output_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
        else:
            getattr(self.lib, self.func_name)(input_ctype, output_ctype, num_threads)

        output = build_tensor(output_ctype, (bs, num_features))
        return output, (input_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        input_ctype, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, bs * num_features)
        grad_input_ctype = (ctypes.c_float * (bs * num_features))(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_input_ctype), None, ctypes.byref(status))
            getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_cl, grad_output_cl, num_features, ctypes.c_int(0), num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_input_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_input_ctype), ctypes.byref(grad_input_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_ctype, output_ctype, num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, num_features))
        return grad_input

# reduce 2nd argument
class Reduce(Module):
    def __init__(self, *args):
        super().__init__(*args)
        func_name, _, target = args
        self.func_name = func_name
        self.target = target
        
    def forward(self, input):
        bs, num_features = input.shape
        num_threads = bs * num_features
        input_ctype = build_ctypes(input, bs * num_features) 
        output_ctype = (ctypes.c_float * (bs))(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(input_ctype), ctypes.byref(input_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            getattr(self.lib, self.func_name)(input_cl, output_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
        else:
            getattr(self.lib, self.func_name)(input_ctype, output_ctype, num_features_ctype, num_threads)
        output = build_tensor(output_ctype, bs)
        return output, (input_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):
        input_ctype, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, bs)
        grad_input_ctype = (ctypes.c_float * (bs * num_features))(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        grad_num_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_input_ctype), None, ctypes.byref(status))
            getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_cl, grad_output_cl, num_features, ctypes.c_int(0), num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_input_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_input_ctype), ctypes.byref(grad_input_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_ctype, output_ctype, num_features_ctype, grad_num_features_ctype, num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, num_features))
        return grad_input

class Sqrt(UnaryModule):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('sqrt_', lib_name, target)

class Sum(Reduce):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('sum_', lib_name, target)

class Mean(Reduce):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('mean_', lib_name, target)

class ReLU(UnaryModule):

    def __init__(self, lib_name='ops', target=TARGET):\
        super().__init__('relu_', lib_name, target)
    
class SiLU(UnaryModule):

    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('silu_', lib_name, target)

class Sigmoid(UnaryModule):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('sigmoid_', lib_name, target)

class BinaryModule(Module):

    def __init__(self, *args):
        super().__init__(*args)
        func_name, _, _ = args
        
    def forward(self, x, y):

        bs, num_features = x.shape
        num_threads = bs * num_features
        x_ctype = build_ctypes(x, bs * num_features) 
        y_ctype = build_ctypes(y, bs * num_features)
        output_ctype = (ctypes.c_float * (bs * num_features))(0)
        getattr(self.lib, self.func_name)(x_ctype, y_ctype, output_ctype, num_threads)
        output = build_tensor(output_ctype, (bs, num_features))
        return output, (x_ctype, y_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        x_ctype, y_ctype, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, bs * num_features)
        grad_x_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_y_ctype = (ctypes.c_float * (bs * num_features))(0)
        getattr(self.lib, f'grad_{self.func_name}')(x_ctype, grad_x_ctype, y_ctype, grad_y_ctype, output_ctype, num_threads)
        grad_x = build_tensor(grad_x_ctype, (bs, num_features))
        grad_y = build_tensor(grad_y_ctype, (bs, num_features))
        return grad_x, grad_y
    
class Add(BinaryModule):

    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('add_', lib_name, target)

class Sub(BinaryModule):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('sub_', lib_name, target)
        
class Mul(BinaryModule):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('multiply_', lib_name, target)

class Div(BinaryModule):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('divide_', lib_name, target)

class MSELoss(BinaryModule):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('mse_', lib_name, target)
        
    def forward(self, x, y):

        bs, num_features = x.shape
        num_threads = bs * num_features
        x_ctype = build_ctypes(x, bs * num_features) 
        y_ctype = build_ctypes(y, bs * num_features)
        bs_ctype = (ctypes.c_int * 1)(bs)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        output_ctype = (ctypes.c_float * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_float()
            cl_utils.cl_check(status.value)
            x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(x_ctype), ctypes.byref(x_ctype), ctypes.byref(status))
            y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(y_ctype), ctypes.byref(y_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            getattr(self.lib, self.func_name)(x_cl, y_cl, output_cl, num_features, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
        else:
            getattr(self.lib, self.func_name)(x_ctype, y_ctype, output_ctype, bs_ctype, num_features_ctype, num_threads)
        output = build_tensor(output_ctype, [])
        return output, (x_ctype, y_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        x_ctype, y_ctype, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, 1)
        grad_x_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_y_ctype = (ctypes.c_float * (bs * num_features))(0)
        bs_ctype = (ctypes.c_int * 1)(bs)
        grad_bs_ctype = (ctypes.c_int * 1)(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        grad_num_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_float()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_x_ctype), None, ctypes.byref(status))
            grad_y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_y_ctype), None, ctypes.byref(status))
            getattr(self.lib, f'grad_{self.func_name}')(x_ctype, grad_x_cl, y_ctype, grad_y_cl, grad_output_cl, bs, ctypes.c_int(0), num_features, ctypes.c_int(0), num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_x_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_x_ctype), ctypes.byref(grad_x_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_y_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_y_ctype), ctypes.byref(grad_y_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(x_ctype, grad_x_ctype, y_ctype, grad_y_ctype, output_ctype, bs_ctype, grad_bs_ctype, 
                                                        num_features_ctype, grad_num_features_ctype, num_threads)
        grad_x = build_tensor(grad_x_ctype, (bs, num_features))
        grad_y = build_tensor(grad_y_ctype, (bs, num_features))
        return grad_x, grad_y

class Linear(Module):
    
    def __init__(self, lib_name='ops', target=TARGET):
        super().__init__('linear_', lib_name, target)
    
    def forward(self, input, weight, bias):

        bs, in_features = input.shape
        out_features = weight.shape[0]
        num_threads = bs * out_features
        input_ctype = build_ctypes(input, bs * in_features)
        weight_ctype = build_ctypes(weight, out_features * in_features)
        bias_ctype = build_ctypes(bias, out_features)
        output_ctype = (ctypes.c_float * (bs * out_features))(0)
        input_features_ctype = (ctypes.c_int * 1)(in_features)
        output_features_ctype = (ctypes.c_int * 1)(out_features)
        if self.target == 'opencl':
            status = ctypes.c_float()
            cl_utils.cl_check(status.value)
            input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(input_ctype), ctypes.byref(input_ctype), ctypes.byref(status))
            weight_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(weight_ctype), ctypes.byref(weight_ctype), ctypes.byref(status))
            bias_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(bias_ctype), ctypes.byref(bias_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            getattr(self.lib, self.func_name)(input_cl, weight_cl, bias_cl, output_cl, input_features_ctype, output_features_ctype, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
        else:
            getattr(self.lib, self.func_name)(input_ctype, weight_ctype, bias_ctype, output_ctype, input_features_ctype, output_features_ctype, num_threads)
        output = build_tensor(output_ctype, (bs, out_features))
        return output, (input_ctype, weight_ctype, bias_ctype, bs, in_features, out_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        input_ctype, weight_ctype, bias_ctype, bs, in_features, out_features, num_threads = input_ctx
        grad_input_ctype = (ctypes.c_float * (bs * in_features))(0)
        grad_weight_ctype = (ctypes.c_float * (out_features * in_features))(0)
        grad_bias_ctype = (ctypes.c_float * (out_features))(0)
        output_ctype = build_ctypes(grad_output, bs * out_features)
        in_features_ctype = (ctypes.c_int * 1)(in_features)
        grad_in_features_ctype = (ctypes.c_int * 1)(0)
        out_features_ctype = (ctypes.c_int * 1)(out_features)
        grad_out_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_float()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_input_ctype), None, ctypes.byref(status))
            grad_weight_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_weight_ctype), None, ctypes.byref(status))
            grad_bias_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_bias_ctype), None, ctypes.byref(status))
            getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_cl, weight_ctype, grad_weight_cl, bias_ctype, grad_bias_cl, grad_output_cl, in_features, out_features, ctypes.c_int(0), num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_input_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_input_ctype), ctypes.byref(grad_input_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_weight_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_weight_ctype), ctypes.byref(grad_weight_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_bias_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_bias_ctype), ctypes.byref(grad_bias_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(input_ctype, grad_input_ctype, weight_ctype, grad_weight_ctype, 
                bias_ctype, grad_bias_ctype, output_ctype, in_features_ctype, grad_in_features_ctype, out_features_ctype, grad_out_features_ctype, 
                num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, in_features))
        grad_weight = build_tensor(grad_weight_ctype, (out_features, in_features))
        grad_bias = build_tensor(grad_bias_ctype, (out_features,))
        return grad_input, grad_weight, grad_bias