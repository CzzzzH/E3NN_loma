import os 
from ctypes import CDLL
from ...loma_public import compiler, cl_utils

class Module:
    
    def __init__(self, lib_name, target='ispc'):
         
        if target == 'ispc':
            if os.path.exists(f'_code/{lib_name}.so'):
                self.lib = CDLL(f'_code/{lib_name}.so')
            else:
                with open('loma_code/basic.py') as f:
                    _, self.lib = compiler.compile(f.read(),
                                            target = 'c',
                                            output_filename = f'_code/{lib_name}')
        elif target == 'opencl':
            # TODO: Implement OpenCL compilation
            raise NotImplementedError('OpenCL compilation is not implemented yet')
        else:
            raise ValueError('target should be either "ispc" or "opencl"')

    def forward():
        pass

    def backward():
        pass
