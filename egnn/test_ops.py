import loma
import torch
import unittest

class TestLomaOperators(unittest.TestCase):
    
    def test_add(self):
        
        loma_add = loma.Add()
        x = torch.rand(4, 10, requires_grad=True)
        y = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_add = loma_add(x, y)
        output_ref_add = x + y
        
        # Backward
        output_ref_add.backward(output_ref_add)
        grad_x_add, grad_y_add = loma_add.backward(output_add)
        grad_x_ref_add, grad_y_ref_add = x.grad, y.grad
        
        assert torch.allclose(output_add, output_ref_add)
        assert torch.allclose(grad_x_add, grad_x_ref_add)
        assert torch.allclose(grad_y_add, grad_y_ref_add)
        
    def test_sub(self):
        
        loma_sub = loma.Sub()
        x = torch.rand(4, 10, requires_grad=True)
        y = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_sub = loma_sub(x, y)
        output_ref_sub = x - y
        
        # Backward
        output_ref_sub.backward(output_ref_sub)
        grad_x_sub, grad_y_sub = loma_sub.backward(output_sub)
        grad_x_ref_sub, grad_y_ref_sub = x.grad, y.grad
        
        assert torch.allclose(output_sub, output_ref_sub)
        assert torch.allclose(grad_x_sub, grad_x_ref_sub)
        assert torch.allclose(grad_y_sub, grad_y_ref_sub)
        
    def test_multiply(self):
        
        loma_multiply = loma.Mul()
        x = torch.rand(4, 10, requires_grad=True)
        y = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_multiply = loma_multiply(x, y)
        output_ref_multiply = x * y
        
        # Backward
        output_ref_multiply.backward(output_ref_multiply)
        grad_x_multiply, grad_y_multiply = loma_multiply.backward(output_multiply)
        grad_x_ref_multiply, grad_y_ref_multiply = x.grad, y.grad
        
        assert torch.allclose(output_multiply, output_ref_multiply)
        assert torch.allclose(grad_x_multiply, grad_x_ref_multiply)
        assert torch.allclose(grad_y_multiply, grad_y_ref_multiply)
        
    def test_divide(self):
        
        loma_divide = loma.Div()
        x = torch.rand(4, 10, requires_grad=True)
        y = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_divide = loma_divide(x, y)
        output_ref_divide = x / y
        
        # Backward
        output_ref_divide.backward(output_ref_divide)
        grad_x_divide, grad_y_divide = loma_divide.backward(output_divide)
        grad_x_ref_divide, grad_y_ref_divide = x.grad, y.grad
        
        assert torch.allclose(output_divide, output_ref_divide)
        assert torch.allclose(grad_x_divide, grad_x_ref_divide)
        assert torch.allclose(grad_y_divide, grad_y_ref_divide)
    
    def test_sqrt(self):
        
        loma_sqrt = loma.Sqrt()
        input_sqrt = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_sqrt = loma_sqrt(input_sqrt)
        output_ref_sqrt = torch.sqrt(input_sqrt)
        
        # Backward
        output_ref_sqrt.backward(output_ref_sqrt)
        grad_input_sqrt = loma_sqrt.backward(output_sqrt)
        grad_input_ref_sqrt = input_sqrt.grad
        
        assert torch.allclose(output_sqrt, output_ref_sqrt)
        assert torch.allclose(grad_input_sqrt, grad_input_ref_sqrt)
    
    def test_sum(self):
        
        loma_sum = loma.Sum()
        input_sum = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_sum = loma_sum(input_sum)
        output_ref_sum = torch.sum(input_sum, dim = 1)
        
        # Backward
        output_ref_sum.backward(output_ref_sum)
        grad_input_sum = loma_sum.backward(output_sum)
        grad_input_ref_sum = input_sum.grad
        
        assert torch.allclose(output_sum, output_ref_sum)
        assert torch.allclose(grad_input_sum, grad_input_ref_sum)

    def test_mean(self):
        
        loma_mean = loma.Mean()
        input_mean = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_mean = loma_mean(input_mean)
        output_ref_mean = torch.mean(input_mean, dim = 1)
        
        # Backward
        output_ref_mean.backward(output_ref_mean)
        grad_input_mean = loma_mean.backward(output_mean)
        grad_input_ref_mean = input_mean.grad
        
        assert torch.allclose(output_mean, output_ref_mean)
        assert torch.allclose(grad_input_mean, grad_input_ref_mean)

    def test_linear(self):
        
        loma_linear = loma.Linear()
        torch_linear = torch.nn.Linear(10, 5)
        
        input_linear = torch.rand(4, 10, requires_grad=True)
        weight = torch.rand(5, 10, requires_grad=True)
        bias = torch.rand(5, requires_grad=True)
        torch_linear.weight.data = weight
        torch_linear.bias.data = bias
        
        # Forward
        output_linear = loma_linear(input_linear, torch_linear.weight, torch_linear.bias)
        output_ref_linear = torch_linear(input_linear)
        
        # Backward
        output_ref_linear.backward(output_ref_linear)
        grad_input_linear = loma_linear.backward(output_linear)
        grad_input_ref_linear = input_linear.grad
        
        assert torch.allclose(output_linear, output_ref_linear)
        assert torch.allclose(grad_input_linear, grad_input_ref_linear)
        
    def test_relu(self):
        
        loma_relu = loma.ReLU()
        torch_relu = torch.nn.ReLU()
        
        input_relu = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_relu = loma_relu(input_relu)
        output_ref_relu = torch_relu(input_relu)
        
        # Backward
        output_ref_relu.backward(output_ref_relu)
        grad_input_relu = loma_relu.backward(output_relu)
        grad_input_ref_relu = input_relu.grad
        
        assert torch.allclose(output_relu, output_ref_relu)
        assert torch.allclose(grad_input_relu, grad_input_ref_relu)
    
    def test_silu(self):
        
        loma_silu = loma.SiLU()
        torch_silu = torch.nn.SiLU()
        
        input_silu = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_silu = loma_silu(input_silu)
        output_ref_silu = torch_silu(input_silu)
        
        # Backward
        output_ref_silu.backward(output_ref_silu)
        grad_input_silu = loma_silu.backward(output_silu)
        grad_input_ref_silu = input_silu.grad
        
        assert torch.allclose(output_silu, output_ref_silu)
        assert torch.allclose(grad_input_silu, grad_input_ref_silu)
    
    def test_sigmoid(self):
        
        loma_sigmoid = loma.Sigmoid()
        torch_sigmoid = torch.nn.Sigmoid()
        
        input_sigmoid = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_sigmoid = loma_sigmoid(input_sigmoid)
        output_ref_sigmoid = torch_sigmoid(input_sigmoid)
        
        # Backward
        output_ref_sigmoid.backward(output_ref_sigmoid)
        grad_input_sigmoid = loma_sigmoid.backward(output_sigmoid)
        grad_input_ref_sigmoid = input_sigmoid.grad
        
        assert torch.allclose(output_sigmoid, output_ref_sigmoid)
        assert torch.allclose(grad_input_sigmoid, grad_input_ref_sigmoid)
    
    def test_mse(self):
        
        loma_mse = loma.MSELoss()
        torch_mse = torch.nn.MSELoss()
        
        x = torch.rand(4, 10, requires_grad=True)
        y = torch.rand(4, 10, requires_grad=True)
        
        # Forward
        output_mse = loma_mse(x, y)
        output_ref_mse = torch_mse(x, y)
        
        # Backward
        output_ref_mse.backward(output_ref_mse)
        grad_x_mse, grad_y_mse = loma_mse.backward(output_mse)
        grad_x_ref_mse, grad_y_ref_mse = x.grad, y.grad
        
        assert torch.allclose(output_mse, output_ref_mse)
        assert torch.allclose(grad_x_mse, grad_x_ref_mse)
        assert torch.allclose(grad_y_mse, grad_y_ref_mse)
        
if __name__ == '__main__':
    
    unittest.main()

    