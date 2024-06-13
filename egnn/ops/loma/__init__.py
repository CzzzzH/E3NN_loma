import os 
import sys

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current, 'loma_public'))
import compiler
import cl_utils
import gpuctypes.opencl as cl

from ctypes import CDLL
from .utils import *

# TARGET = 'ispc'
TARGET = 'opencl'
LIB_NAME = 'ops'
DEBUG = False

if TARGET == 'ispc':
    if os.path.exists(f'{current}/_code/{LIB_NAME}.so'):
        lib = CDLL(f'{current}/_code/{LIB_NAME}.so')
    else:
        with open(f'{current}/loma_code/{LIB_NAME}.py') as f:
            _, lib = compiler.compile(f.read(),
                                    target = 'ispc',
                                    output_filename = f'{current}/_code/{LIB_NAME}')
elif TARGET == 'c':
    if os.path.exists(f'{current}/_code/{LIB_NAME}.so'):
        lib = CDLL(f'{current}/_code/{LIB_NAME}.so')
    else:
        with open(f'{current}/loma_code/{LIB_NAME}.py') as f:
            _, lib = compiler.compile(f.read(),
                                    target = 'c',
                                    output_filename = f'{current}/_code/{LIB_NAME}')
elif TARGET == 'opencl':
    cl_ctx, cl_device, cl_cmd_queue = cl_utils.create_context()
    with open(f'{current}/loma_code/{LIB_NAME}.py') as f:
        _, lib = compiler.compile(f.read(),
                                target = 'opencl',
                                opencl_context = cl_ctx,
                                opencl_device = cl_device,
                                opencl_command_queue = cl_cmd_queue)
else:
    raise ValueError('target should be either "c" or "ispc" or "opencl"')

class MemoryManager:
    
    def __init__(self):
        self.buffers = []
    
    def add_buffers(self, buffers):
        self.buffers.extend(buffers)
        
    def release_buffers(self):
        for buf in self.buffers:
            cl.clReleaseMemObject(buf)
        self.buffers = []

memory_manager = MemoryManager()

class Module:
    
    def __init__(self, *args):
        
        func_name, target = args
        self.func_name = func_name
        self.lib = lib
        self.target = target
        if target == 'opencl':
            self.cl_ctx = cl_ctx
            self.cl_device = cl_device
            self.cl_cmd_queue = cl_cmd_queue
            self.cl_mem = memory_manager
        
    def forward(self, *args):
        raise NotImplementedError('forward method is not implemented')

    def backward(self, *args):
        raise NotImplementedError('backward method is not implemented')

    def __call__(self, *args):
        return self.forward(*args)

class UnaryModule(Module):

    def __init__(self, *args):
        super().__init__(*args)
        
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
            self.cl_mem.add_buffers([input_cl, output_cl])
            getattr(self.lib, self.func_name)(input_cl, output_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
            output = build_tensor(output_ctype, (bs, num_features))
            return output, (input_cl, bs, num_features, num_threads)
        else:
            getattr(self.lib, self.func_name)(input_ctype, output_ctype, num_threads)
            output = build_tensor(output_ctype, (bs, num_features))
            return output, (input_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):
        
        input_tensor, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, bs * num_features)
        grad_input_ctype = (ctypes.c_float * (bs * num_features))(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_input_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([grad_output_cl, grad_input_cl])
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_cl, grad_output_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_input_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_input_ctype), ctypes.byref(grad_input_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_ctype, output_ctype, num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, num_features)) # output
        return grad_input

# reduce 1st argument, dim=0
class Reduce(Module):
    def __init__(self, *args):
        super().__init__(*args)
        
    def forward(self, input):
        bs, num_features = input.shape
        num_threads = bs * num_features
        input_ctype = build_ctypes(input, bs * num_features) 
        output_ctype = (ctypes.c_float * (num_features))(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(input_ctype), ctypes.byref(input_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            self.cl_mem.add_buffers([input_cl, output_cl, num_features_cl])
            getattr(self.lib, self.func_name)(input_cl, output_cl, num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
            output = build_tensor(output_ctype, (1, num_features))
            return output, (input_cl, bs, num_features, num_threads)
        else:
            getattr(self.lib, self.func_name)(input_ctype, output_ctype, num_features_ctype, num_threads)
            output = build_tensor(output_ctype, (1, num_features))
            return output, (input_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):
        input_tensor, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, num_features)
        grad_input_ctype = (ctypes.c_float * (bs * num_features))(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        grad_num_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_input_ctype), None, ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            grad_num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_num_features_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([output_cl, grad_input_cl, num_features_cl, grad_num_features_cl])
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_cl, output_cl, num_features_cl, grad_num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_input_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_input_ctype), ctypes.byref(grad_input_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_ctype, output_ctype, num_features_ctype, grad_num_features_ctype, num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, num_features))
        return grad_input

# sum aggregation
class SumAggr(Module):
    def __init__(self, target=TARGET):
        super().__init__('sum_aggr_', target)
        
    def forward(self, input, index):
        bs, num_features = input.shape
        index = index.int()
        # get the max index
        out_size = index.max().item() + 1
        num_threads = bs * num_features
        input_ctype = build_ctypes(input, bs * num_features) 
        index_ctype = build_ctypes(index, bs)
        output_ctype = (ctypes.c_float * (out_size * num_features))(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(input_ctype), ctypes.byref(input_ctype), ctypes.byref(status))
            index_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(index_ctype), ctypes.byref(index_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            self.cl_mem.add_buffers([input_cl, index_cl, output_cl, num_features_cl])
            getattr(self.lib, self.func_name)(input_cl, index_cl, output_cl, num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
            output = build_tensor(output_ctype, (out_size, num_features))
            return output, (input_cl, index_cl, bs, num_features, num_threads)
        else:
            getattr(self.lib, self.func_name)(input_ctype, index_ctype, output_ctype, num_features_ctype, num_threads)
            output = build_tensor(output_ctype, (out_size, num_features))
            return output, (input_ctype, index_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):
        input_tensor, index_tensor, bs, num_features, num_threads = input_ctx
        # get size of the output directly from the grad_output shape
        size_all = grad_output.shape[0] * grad_output.shape[1]
        output_ctype = build_ctypes(grad_output, size_all)
        grad_input_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_index_ctype = (ctypes.c_int * bs)(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        grad_num_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_input_ctype), None, ctypes.byref(status))
            grad_index_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_index_ctype), None, ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            grad_num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_num_features_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([output_cl, grad_input_cl, grad_index_cl, num_features_cl, grad_num_features_cl])
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_cl, index_tensor, grad_index_cl, output_cl, num_features_cl, grad_num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_input_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_input_ctype), ctypes.byref(grad_input_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_ctype, index_tensor, grad_index_ctype, output_ctype, num_features_ctype, grad_num_features_ctype, num_threads)
        grad_input = build_tensor(grad_input_ctype, (bs, num_features))
        return grad_input, None

class Sqrt(UnaryModule):
    
    def __init__(self, target=TARGET):
        super().__init__('sqrt_', target)

class Sum(Reduce):
    
    def __init__(self, target=TARGET):
        super().__init__('sum_', target)

class ReLU(UnaryModule):

    def __init__(self, target=TARGET):\
        super().__init__('relu_', target)
    
class SiLU(UnaryModule):

    def __init__(self, target=TARGET):
        super().__init__('silu_', target)

class Sigmoid(UnaryModule):
    
    def __init__(self, target=TARGET):
        super().__init__('sigmoid_', target)
        
class BinaryModule(Module):

    def __init__(self, *args):
        super().__init__(*args)
        
    def forward(self, x, y):

        bs, num_features = x.shape
        num_threads = bs * num_features
        x_ctype = build_ctypes(x, bs * num_features) 
        y_ctype = build_ctypes(y, bs * num_features)
        output_ctype = (ctypes.c_float * (bs * num_features))(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(x_ctype), ctypes.byref(x_ctype), ctypes.byref(status))
            y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(y_ctype), ctypes.byref(y_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([x_cl, y_cl, output_cl])
            getattr(self.lib, self.func_name)(x_cl, y_cl, output_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
            output = build_tensor(output_ctype, (bs, num_features))
            return output, (x_cl, y_cl, bs, num_features, num_threads)
        else:
            getattr(self.lib, self.func_name)(x_ctype, y_ctype, output_ctype, num_threads)
            output = build_tensor(output_ctype, (bs, num_features))
            return output, (x_ctype, y_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        x_tensor, y_tensor, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, bs * num_features)
        grad_x_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_y_ctype = (ctypes.c_float * (bs * num_features))(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_x_ctype), None, ctypes.byref(status))
            grad_y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_y_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([grad_output_cl, grad_x_cl, grad_y_cl])
            getattr(self.lib, f'grad_{self.func_name}')(x_tensor, grad_x_cl, y_tensor, grad_y_cl, grad_output_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_x_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_x_ctype), ctypes.byref(grad_x_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_y_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_y_ctype), ctypes.byref(grad_y_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(x_tensor, grad_x_ctype, y_tensor, grad_y_ctype, output_ctype, num_threads)
        grad_x = build_tensor(grad_x_ctype, (bs, num_features))
        grad_y = build_tensor(grad_y_ctype, (bs, num_features))
        return grad_x, grad_y

class BinaryBroadcast(Module):
    
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, y):
        bs, num_features = x.shape
        num_threads = bs * num_features
        x_ctype = build_ctypes(x, bs * num_features) 
        y_ctype = build_ctypes(y, 1 * num_features)
        output_ctype = (ctypes.c_float * (bs * num_features))(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(x_ctype), ctypes.byref(x_ctype), ctypes.byref(status))
            y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(y_ctype), ctypes.byref(y_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            self.cl_mem.add_buffers([x_cl, y_cl, output_cl, num_features_cl])
            getattr(self.lib, self.func_name)(x_cl, y_cl, output_cl, num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
            output = build_tensor(output_ctype, (bs, num_features))
            return output, (x_cl, y_cl, bs, num_features, num_threads)
        else:
            getattr(self.lib, self.func_name)(x_ctype, y_ctype, output_ctype, num_threads)
            output = build_tensor(output_ctype, (bs, num_features))
            return output, (x_ctype, y_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        x_tensor, y_tensor, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, bs * num_features)
        grad_x_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_y_ctype = (ctypes.c_float * (1 * num_features))(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        grad_num_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_x_ctype), None, ctypes.byref(status))
            grad_y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_y_ctype), None, ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            grad_num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_num_features_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([grad_output_cl, grad_x_cl, grad_y_cl, num_features_cl, grad_num_features_cl])
            getattr(self.lib, f'grad_{self.func_name}')(x_tensor, grad_x_cl, y_tensor, grad_y_cl, grad_output_cl, num_features_cl, grad_num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_x_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_x_ctype), ctypes.byref(grad_x_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_y_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_y_ctype), ctypes.byref(grad_y_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(x_tensor, grad_x_ctype, y_tensor, grad_y_ctype, output_ctype, num_features_ctype, grad_num_features_ctype, num_threads)
        grad_x = build_tensor(grad_x_ctype, (bs, num_features))
        grad_y = build_tensor(grad_y_ctype, (1, num_features))
        return grad_x, grad_y    

class BinaryLoss(Module):
    
    def __init__(self, *args):
        super().__init__(*args)
        
    def forward(self, x, y):

        bs, num_features = x.shape
        num_threads = bs * num_features
        x_ctype = build_ctypes(x, bs * num_features) 
        y_ctype = build_ctypes(y, bs * num_features)
        bs_ctype = (ctypes.c_int * 1)(bs)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        output_ctype = (ctypes.c_float * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(x_ctype), ctypes.byref(x_ctype), ctypes.byref(status))
            y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(y_ctype), ctypes.byref(y_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            bs_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(bs_ctype), ctypes.byref(bs_ctype), ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            self.cl_mem.add_buffers([x_cl, y_cl, output_cl, bs_cl, num_features_cl])
            getattr(self.lib, self.func_name)(x_cl, y_cl, output_cl, bs_cl, num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
            output = build_tensor(output_ctype, [])
            return output, (x_cl, y_cl, bs, num_features, num_threads)
        else:
            getattr(self.lib, self.func_name)(x_ctype, y_ctype, output_ctype, bs_ctype, num_features_ctype, num_threads)
            output = build_tensor(output_ctype, [])
            return output, (x_ctype, y_ctype, bs, num_features, num_threads)

    def backward(self, grad_output, *input_ctx):
        x_tensor, y_tensor, bs, num_features, num_threads = input_ctx
        output_ctype = build_ctypes(grad_output, 1)
        grad_x_ctype = (ctypes.c_float * (bs * num_features))(0)
        grad_y_ctype = (ctypes.c_float * (bs * num_features))(0)
        bs_ctype = (ctypes.c_int * 1)(bs)
        grad_bs_ctype = (ctypes.c_int * 1)(0)
        num_features_ctype = (ctypes.c_int * 1)(num_features)
        grad_num_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_x_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_x_ctype), None, ctypes.byref(status))
            grad_y_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_y_ctype), None, ctypes.byref(status))
            bs_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(bs_ctype), ctypes.byref(bs_ctype), ctypes.byref(status))
            grad_bs_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_bs_ctype), None, ctypes.byref(status))
            num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(num_features_ctype), ctypes.byref(num_features_ctype), ctypes.byref(status))
            grad_num_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_num_features_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([grad_output_cl, grad_x_cl, grad_y_cl, bs_cl, grad_bs_cl, num_features_cl, grad_num_features_cl])
            getattr(self.lib, f'grad_{self.func_name}')(x_tensor, grad_x_cl, y_tensor, grad_y_cl, grad_output_cl, bs_cl, grad_bs_cl, num_features_cl, grad_num_features_cl, num_threads)
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_x_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_x_ctype), ctypes.byref(grad_x_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_y_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_y_ctype), ctypes.byref(grad_y_ctype), 0, None, None)
        else:
            getattr(self.lib, f'grad_{self.func_name}')(x_tensor, grad_x_ctype, y_tensor, grad_y_ctype, output_ctype, bs_ctype, grad_bs_ctype, 
                                                        num_features_ctype, grad_num_features_ctype, num_threads)
        grad_x = build_tensor(grad_x_ctype, (bs, num_features))
        grad_y = build_tensor(grad_y_ctype, (bs, num_features))
        return grad_x, grad_y

class Add(BinaryModule):

    def __init__(self, target=TARGET):
        super().__init__('add_', target)

class Sub(BinaryModule):
    
    def __init__(self, target=TARGET):
        super().__init__('sub_',  target)
        
class Mul(BinaryModule):
    
    def __init__(self, target=TARGET):
        super().__init__('multiply_', target)

class Div(BinaryModule):
    
    def __init__(self, target=TARGET):
        super().__init__('divide_', target)

class AddBroadcast(BinaryBroadcast):
    
    def __init__(self, target=TARGET):
        super().__init__('add_broadcast_', target)
        
class MulBroadcast(BinaryBroadcast):
    
    def __init__(self, target=TARGET):
        super().__init__('multiply_broadcast_', target)

class MAELoss(BinaryLoss):
    
    def __init__(self, target=TARGET):
        super().__init__('mae_', target)
        
class MSELoss(BinaryLoss):
    
    def __init__(self, target=TARGET):
        super().__init__('mse_', target)

class Linear(Module):
    
    def __init__(self, target=TARGET):
        super().__init__('linear_', target)
    
    def forward(self, input, weight, bias):
        
        # Check time
        if DEBUG:
            import time
            start = time.time()
        
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
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(input_ctype), ctypes.byref(input_ctype), ctypes.byref(status))
            weight_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(weight_ctype), ctypes.byref(weight_ctype), ctypes.byref(status))
            bias_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(bias_ctype), ctypes.byref(bias_ctype), ctypes.byref(status))
            output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(output_ctype), None, ctypes.byref(status))
            input_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(input_features_ctype), ctypes.byref(input_features_ctype), ctypes.byref(status))
            output_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_features_ctype), ctypes.byref(output_features_ctype), ctypes.byref(status))
            self.cl_mem.add_buffers([input_cl, weight_cl, bias_cl, output_cl, input_features_cl, output_features_cl])
            if DEBUG: print(f"Time to create buffers: {time.time() - start}")
            getattr(self.lib, self.func_name)(input_cl, weight_cl, bias_cl, output_cl, input_features_cl, output_features_cl, num_threads)
            if DEBUG: print(f"Time to run kernel: {time.time() - start}")
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, output_cl, cl.CL_TRUE, 0, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), 0, None, None)
            output = build_tensor(output_ctype, (bs, out_features))
            if DEBUG: print(f"Time to output: {time.time() - start}\n")
            return output, (input_cl, weight_cl, bias_cl, bs, in_features, out_features, num_threads)
        else:
            if DEBUG: print(f"Time to create buffers: {time.time() - start}")
            getattr(self.lib, self.func_name)(input_ctype, weight_ctype, bias_ctype, output_ctype, input_features_ctype, output_features_ctype, num_threads)
            if DEBUG: print(f"Time to run kernel: {time.time() - start}")
            output = build_tensor(output_ctype, (bs, out_features))
            if DEBUG: print(f"Time to output: {time.time() - start}\n")
            return output, (input_ctype, weight_ctype, bias_ctype, bs, in_features, out_features, num_threads)

    def backward(self, grad_output, *input_ctx):

        import time
        start = time.time()

        input_tensor, weight_tensor, bias_tensor, bs, in_features, out_features, num_threads = input_ctx
        grad_input_ctype = (ctypes.c_float * (bs * in_features))(0)
        grad_weight_ctype = (ctypes.c_float * (out_features * in_features))(0)
        grad_bias_ctype = (ctypes.c_float * (out_features))(0)
        output_ctype = build_ctypes(grad_output, bs * out_features)
        in_features_ctype = (ctypes.c_int * 1)(in_features)
        grad_in_features_ctype = (ctypes.c_int * 1)(0)
        out_features_ctype = (ctypes.c_int * 1)(out_features)
        grad_out_features_ctype = (ctypes.c_int * 1)(0)
        if self.target == 'opencl':
            status = ctypes.c_int32()
            cl_utils.cl_check(status.value)
            grad_output_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(output_ctype), ctypes.byref(output_ctype), ctypes.byref(status))
            grad_input_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_input_ctype), None, ctypes.byref(status))
            grad_weight_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_weight_ctype), None, ctypes.byref(status))
            grad_bias_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_bias_ctype), None, ctypes.byref(status))
            in_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(in_features_ctype), ctypes.byref(in_features_ctype), ctypes.byref(status))
            grad_in_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_in_features_ctype), None, ctypes.byref(status))
            out_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, ctypes.sizeof(out_features_ctype), ctypes.byref(out_features_ctype), ctypes.byref(status))
            grad_out_features_cl = cl.clCreateBuffer(self.cl_ctx, cl.CL_MEM_WRITE_ONLY, ctypes.sizeof(grad_out_features_ctype), None, ctypes.byref(status))
            self.cl_mem.add_buffers([grad_output_cl, grad_input_cl, grad_weight_cl, grad_bias_cl, in_features_cl, grad_in_features_cl, out_features_cl, grad_out_features_cl])
            if DEBUG: print(f"Time to create buffers (backward): {time.time() - start}")
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_cl, weight_tensor, grad_weight_cl, bias_tensor, grad_bias_cl, grad_output_cl, in_features_cl, grad_in_features_cl, out_features_cl, grad_out_features_cl, num_threads)
            if DEBUG: print(f"Time to run kernel (backward): {time.time() - start}")
            cl.clFinish(self.cl_cmd_queue)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_input_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_input_ctype), ctypes.byref(grad_input_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_weight_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_weight_ctype), ctypes.byref(grad_weight_ctype), 0, None, None)
            cl.clEnqueueReadBuffer(self.cl_cmd_queue, grad_bias_cl, cl.CL_TRUE, 0, ctypes.sizeof(grad_bias_ctype), ctypes.byref(grad_bias_ctype), 0, None, None)
        else:
            if DEBUG: print(f"Time to create buffers (backward): {time.time() - start}")
            getattr(self.lib, f'grad_{self.func_name}')(input_tensor, grad_input_ctype, weight_tensor, grad_weight_ctype, 
                bias_tensor, grad_bias_ctype, output_ctype, in_features_ctype, grad_in_features_ctype, out_features_ctype, grad_out_features_ctype, 
                num_threads)
            if DEBUG: print(f"Time to run kernel (backward): {time.time() - start}")
        grad_input = build_tensor(grad_input_ctype, (bs, in_features))
        grad_weight = build_tensor(grad_weight_ctype, (out_features, in_features))
        grad_bias = build_tensor(grad_bias_ctype, (out_features,))
        if DEBUG: print(f"Time to output (backward): {time.time() - start}\n")
        return grad_input, grad_weight, grad_bias