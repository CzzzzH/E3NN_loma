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
    
    # Test vector_add
    v1 = torch.rand(10)
    v2 = torch.rand(10)
    v1_ctype = build_ctypes(v1, 10)
    v2_ctype = build_ctypes(v2, 10)
    v3_ctype = build_ctypes(torch.zeros(10), 10)
    lib.vector_add(v1_ctype, v2_ctype, v3_ctype, 10)
    v3_out = build_tensor(v3_ctype, (10,))
    v3_ref = v1 + v2
    check_res(v3_out, v3_ref, "vector_add")
    
    # Test linear
    batch_size = 4
    in_features = 10
    out_features = 5
    input = torch.rand(batch_size, in_features)
    weight = torch.rand(out_features, in_features)
    bias = torch.rand(out_features)
    output = torch.zeros(batch_size, out_features)
    torch_linear = torch.nn.Linear(in_features, out_features)
    torch_linear.weight.data = weight
    torch_linear.bias.data = bias
    
    input_ctype = build_ctypes(input, in_features * batch_size)
    weight_ctype = build_ctypes(weight, in_features * out_features)
    bias_ctype = build_ctypes(bias, out_features)
    output_ctype = build_ctypes(output, out_features * batch_size)
    num_threads = batch_size * out_features
    lib.linear(input_ctype, weight_ctype, bias_ctype, output_ctype, in_features, out_features, num_threads)
    output_out = build_tensor(output_ctype, (batch_size, out_features))
    output_ref = torch_linear(input).detach()
    check_res(output_out, output_ref, "linear")
    
    # Test silu
    batch_size = 4
    out_features = 20
    input = torch.rand(batch_size, out_features)
    output = torch.zeros(batch_size, out_features)
    torch_silu = torch.nn.SiLU()
    input_ctype = build_ctypes(input, out_features * batch_size)
    output_ctype = build_ctypes(output, out_features * batch_size)
    num_threads = batch_size * out_features
    lib.silu(input_ctype, output_ctype, out_features, num_threads)
    output_out = build_tensor(output_ctype, (batch_size, out_features))
    output_ref = torch_silu(input).detach()
    check_res(output_out, output_ref, "silu")

if __name__ == '__main__':
    
    test_basic_ispc()
    
