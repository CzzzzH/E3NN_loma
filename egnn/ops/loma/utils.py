import ctypes
import numpy as np
import torch

def check_res(a, b, test_name):
    if a.shape != b.shape:
        print(f"Expected Shape: {b.shape}")
        print(f"Got Shape: {a.shape}")
        print(f"Test {test_name} failed: Shapes do not match")
        assert False
    if not torch.allclose(a, b):
        print(f"Expected: {b}")
        print(f"Got: {a}")
        print(f"Test {test_name} failed: Wrong result")
        assert False
    print(f"Test {test_name} passed!")
    print()
    
def build_ctypes(tensor, length):
    if tensor.requires_grad:
        tensor = tensor.detach()
    np_array = tensor.flatten().numpy()
    return np.ctypeslib.as_ctypes(np_array)

def build_tensor(c_arr, shape):
    res = torch.frombuffer(c_arr, dtype=torch.float32)
    res = res.reshape(shape)
    return res

def build_numpys(c_struct):
    res = np.ctypeslib.as_array(c_struct)
    return res