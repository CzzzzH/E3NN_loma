import ctypes
from ctypes.util import find_library
l = find_library('OpenCL')
assert(l is not None)
print(ctypes.CDLL(l))