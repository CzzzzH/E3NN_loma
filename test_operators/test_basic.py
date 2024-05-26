import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'loma_public'))
import compiler
import ctypes

import numpy as np
from utils import *
np.random.seed(0)

def test_basic_ispc():
    
    with open('loma_code/basic.py') as f:
        struct, lib = compiler.compile(f.read(),
                                  target = 'ispc',
                                  output_filename = '_code/basic')
    
    # Test vector_add
    v1 = np.random.rand(10)
    v2 = np.random.rand(10)
    v1_ctype = build_ctypes([v1, 10], struct, 'Vector')
    v2_ctype = build_ctypes([v2, 10], struct, 'Vector')
    v3_ctype = build_ctypes([np.zeros_like(v1), 10], struct, 'Vector')
    lib.vector_add(v1_ctype, v2_ctype, v3_ctype)
    v3_out = build_numpys(v3_ctype, 'Vector')
    v3_ref = v1 + v2
    check_res(v3_out, v3_ref, "vector_add")
    
    # a = np.random.rand(5, 3)
    # b = np.random.rand(3, 2)
    # c = a @ b
    # print(c.shape)
    # a_arr = a.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # b_arr = b.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # c = a @ b
    # print(c.flatten)

if __name__ == '__main__':
    
    test_basic_ispc()
    
