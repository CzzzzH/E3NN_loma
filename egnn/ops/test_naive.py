import loma
import torch
import unittest

class TestLomaOperators(unittest.TestCase):
   
    def test_sqrt(self):
        
        loma_sqrt = loma.Sqrt()
        input_sqrt = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_sqrt, input_ctx = loma_sqrt(input_sqrt)
        output_ref_sqrt = torch.sqrt(input_sqrt)
        
        # Backward
        output_ref_sqrt.backward(output_ref_sqrt)
        grad_input_sqrt = loma_sqrt.backward(output_sqrt, *input_ctx)
        grad_input_ref_sqrt = input_sqrt.grad
        
        assert torch.allclose(output_sqrt, output_ref_sqrt)
        assert torch.allclose(grad_input_sqrt, grad_input_ref_sqrt)

     
if __name__ == '__main__':
    
    # only test sqrt
    test = TestLomaOperators()
    test.test_sqrt()

    