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
    v1_ctype = build_ctypes([v1, 10], struct, None)
    v2_ctype = build_ctypes([v2, 10], struct, None)
    v3_ctype = build_ctypes([np.zeros_like(v1), 10], struct, None)
    lib.vector_add(v1_ctype, v2_ctype, v3_ctype, 10)
    v3_out = build_numpys(v3_ctype, None)
    v3_ref = v1 + v2
    check_res(v3_out, v3_ref, "vector_add")
    
    # Test matrix_mul
    m1 = np.random.rand(3, 10)
    m2 = np.random.rand(10, 3)
    m1_ctype = build_ctypes([m1, 30], struct, None)
    m2_ctype = build_ctypes([m2, 30], struct, None)
    m3_ctype = build_ctypes([np.zeros((3, 3)), 9], struct, None)
    lib.matrix_mul(m1_ctype, m2_ctype, m3_ctype, 3, 10, 3, 10)
    m3_out = build_numpys(m3_ctype, None).reshape(3, 3)
    m3_ref = m1 @ m2
    check_res(m3_out, m3_ref, "matrix_mul")

if __name__ == '__main__':
    
    test_basic_ispc()
    
