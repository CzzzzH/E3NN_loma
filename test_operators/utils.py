import numpy as np

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