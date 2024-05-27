import ctypes
import numpy as np
from copy import copy

def check_res(a, b, test_name):
    if a.shape != b.shape:
        print(f"Expected Shape: {b.shape}")
        print(f"Got Shape: {a.shape}")
        print(f"Test {test_name} failed: Shapes do not match")
        assert False
    print(f"Expected: {b}")
    print(f"Got: {a}")
    if not np.allclose(a, b):
        print(f"Test {test_name} failed: Wrong result")
        assert False
    print(f"Test {test_name} passed!")
    print()
    
def build_ctypes(elements, struct_dict, struct_name):
    
    if struct_name == None:
        return (ctypes.c_float * elements[1])(*(elements[0].astype(np.float32).flatten()))
    
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
    elif struct_name == None:
        res = np.ctypeslib.as_array(c_struct)
    else:
        raise ValueError(f"Unknown struct name: {struct_name}")
    return res