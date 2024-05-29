import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'loma_public'))
import compiler

import torch
from utils import *
torch.manual_seed(0)

def test_basic_ispc():
    
    with open('loma_code/basic.py') as f:
        struct, lib = compiler.compile(f.read(),
                                  target = 'ispc',
                                  output_filename = '_code/basic')
    
    # Test add
    batch_size = 4
    in_features = out_features = 10
    in_shape = (batch_size, in_features)
    out_shape = (batch_size, out_features)
    num_threads = batch_size * out_features
    v1 = torch.rand(in_shape, requires_grad=True)
    v2 = torch.rand(in_shape, requires_grad=True)
    v1_ctype = build_ctypes(v1, batch_size * in_features)
    v1_grad_ctype = build_ctypes(torch.zeros_like(v1), batch_size * in_features)
    v2_ctype = build_ctypes(v2, batch_size * in_features)
    v2_grad_ctype = build_ctypes(torch.zeros_like(v2), batch_size * in_features)
    v3_ctype = build_ctypes(torch.zeros(out_shape), out_features * batch_size)
    lib.add(v1_ctype, v2_ctype, v3_ctype, in_features, num_threads)
    v3_out = build_tensor(v3_ctype, out_shape)
    v3_ref = v1 + v2
    check_res(v3_out, v3_ref, "add")

    # Test grad_add
    v3_ref.backward(v3_ref)
    lib.grad_add(v1_ctype, v1_grad_ctype, v2_ctype, v2_grad_ctype, v3_ctype, in_features, ctypes.c_int(0), num_threads)
    v1_grad = build_tensor(v1_grad_ctype, in_shape)
    v2_grad = build_tensor(v2_grad_ctype, in_shape)
    check_res(v1_grad, v1.grad, "grad_add_1")
    check_res(v2_grad, v2.grad, "grad_add_2")

    # Test sum
    batch_size = 4
    in_features = 10
    out_features = 1
    in_shape = (batch_size, in_features)
    out_shape = (batch_size,)
    num_threads = batch_size * in_features
    input = torch.rand(in_shape, requires_grad=True)
    input_ctype = build_ctypes(input, batch_size * in_features)
    input_grad_ctype = build_ctypes(torch.zeros_like(input), batch_size * in_features)
    output_ctype = build_ctypes(torch.zeros(out_shape), batch_size * out_features)
    lib.sum(input_ctype, output_ctype, in_features, num_threads)
    output_out = build_tensor(output_ctype, out_shape)
    output_ref = torch.sum(input, dim=1)
    check_res(output_out, output_ref, "sum")

    # Test grad_sum
    output_ref.backward(output_ref)
    lib.grad_sum(input_ctype, input_grad_ctype, output_ctype, in_features, ctypes.c_int(0), num_threads)
    input_grad = build_tensor(input_grad_ctype, in_shape)
    check_res(input_grad, input.grad, "grad_sum")

    # Test linear
    batch_size = 4
    in_features = 10
    out_features = 5
    in_shape = (batch_size, in_features)
    out_shape = (batch_size, out_features)
    num_threads = batch_size * out_features
    input = torch.rand(in_shape, requires_grad=True)
    weight = torch.rand((out_features, in_features), requires_grad=True)
    bias = torch.rand((out_features,), requires_grad=True)
    torch_linear = torch.nn.Linear(in_features, out_features)
    torch_linear.weight.data = weight
    torch_linear.bias.data = bias
    
    input_ctype = build_ctypes(input, in_features * batch_size)
    input_grad_ctype = build_ctypes(torch.zeros_like(input), batch_size * in_features)
    weight_ctype = build_ctypes(weight, in_features * out_features)
    weight_grad_ctype = build_ctypes(torch.zeros_like(weight), out_features * in_features)
    bias_ctype = build_ctypes(bias, out_features)
    bias_grad_ctype = build_ctypes(torch.zeros_like(bias), out_features)
    output_ctype = build_ctypes(torch.zeros(out_shape), batch_size * out_features)
    lib.linear(input_ctype, weight_ctype, bias_ctype, output_ctype, in_features, out_features, num_threads)
    output_out = build_tensor(output_ctype, (batch_size, out_features))
    output_ref = torch_linear(input)
    check_res(output_out, output_ref, "linear")
    
    # Test grad_linear
    output_ref.backward(output_ref)
    lib.grad_linear(input_ctype, input_grad_ctype, weight_ctype, weight_grad_ctype, bias_ctype, bias_grad_ctype,
                    output_ctype, in_features, ctypes.c_int(0), out_features, ctypes.c_int(0), num_threads)
    input_grad = build_tensor(input_grad_ctype, in_shape)
    weight_grad = build_tensor(weight_grad_ctype, (out_features, in_features))
    bias_grad = build_tensor(bias_grad_ctype, (out_features,))
    check_res(input_grad, input.grad, "grad_linear_input")
    check_res(weight_grad, torch_linear.weight.grad, "grad_linear_weight")
    check_res(bias_grad, torch_linear.bias.grad, "grad_linear_bias")

    # Test relu
    batch_size = 4
    in_features = out_features = 10
    in_shape = (batch_size, in_features)
    out_shape = (batch_size, out_features)
    num_threads = batch_size * out_features
    input = torch.rand(in_shape, requires_grad=True)
    output = torch.zeros(out_shape)
    torch_relu = torch.nn.ReLU()
    input_ctype = build_ctypes(input, batch_size * in_features)
    input_grad_ctype = build_ctypes(torch.zeros_like(input), batch_size * in_features)
    output_ctype = build_ctypes(output, batch_size * out_features)
    lib.relu(input_ctype, output_ctype, out_features, num_threads)
    output_out = build_tensor(output_ctype, out_shape)
    output_ref = torch_relu(input)
    check_res(output_out, output_ref, "relu")

    # Test grad_relu
    output_ref.backward(output_ref)
    lib.grad_relu(input_ctype, input_grad_ctype, output_ctype, out_features, ctypes.c_int(0), num_threads)
    input_grad = build_tensor(input_grad_ctype, out_shape)
    check_res(input_grad, input.grad, "grad_relu")

    # Test silu
    batch_size = 4
    in_features = out_features = 10
    in_shape = (batch_size, in_features)
    out_shape = (batch_size, out_features)
    num_threads = batch_size * out_features
    input = torch.rand(in_shape, requires_grad=True)
    output = torch.zeros(out_shape)
    torch_silu = torch.nn.SiLU()
    input_ctype = build_ctypes(input, batch_size * in_features)
    input_grad_ctype = build_ctypes(torch.zeros_like(input), batch_size * in_features)
    output_ctype = build_ctypes(output, batch_size * out_features)
    lib.silu(input_ctype, output_ctype, out_features, num_threads)
    output_out = build_tensor(output_ctype, out_shape)
    output_ref = torch_silu(input)
    check_res(output_out, output_ref, "silu")

    # Test grad_silu
    output_ref.backward(output_ref)
    lib.grad_silu(input_ctype, input_grad_ctype, output_ctype, out_features, ctypes.c_int(0), num_threads)
    input_grad = build_tensor(input_grad_ctype, out_shape)
    check_res(input_grad, input.grad, "grad_silu")

if __name__ == '__main__':

    test_basic_ispc()
    
