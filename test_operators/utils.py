import ctypes
import numpy as np
from copy import copy

def check_res(a, b, test_name):
    if a.shape != b.shape:
        print(f"Test {test_name} failed: Shapes do not match")
        assert False
    if not np.allclose(a, b):
        print(f"Test {test_name} failed: Wrong result")
        print(f"Expected: {b.flatten()}")
        print(f"Got: {a.flatten()}")
        assert False
    print(f"Test {test_name} passed")
    
def build_ctypes(elements, struct_dict, struct_name):
    ctype_struct = struct_dict[struct_name]
    if struct_name == "Vector":
        val = (ctypes.c_float * elements[1])(*(elements[0].astype(np.float32).flatten()))
        arr = ctype_struct(val, elements[1])
    else:
        raise ValueError(f"Unknown struct name: {struct_name}")
    return arr

def build_numpys(c_struct, struct_name):
    if struct_name == "Vector":
        res = np.ctypeslib.as_array(c_struct.data, shape=(c_struct.size,))
    else:
        raise ValueError(f"Unknown struct name: {struct_name}")
    return res