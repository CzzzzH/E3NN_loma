import loma
import torch

if __name__ == '__main__':
    
    linear = loma.Linear()
    torch_linear = torch.nn.Linear(10, 5)
    silu = loma.SiLU()
    torch_silu = torch.nn.SiLU()

    
    input_linear = torch.rand(4, 10, requires_grad=True)
    weight = torch.rand(5, 10, requires_grad=True)
    bias = torch.rand(5, requires_grad=True)
    torch_linear.weight.data = weight
    torch_linear.bias.data = bias
    input_silu = torch.rand(4, 10, requires_grad=True)

    output_linear = linear(input_linear, torch_linear.weight, torch_linear.bias)
    output_silu = silu(input_silu)
    output_ref_linear = torch_linear(input_linear)
    output_ref_silu = torch_silu(input_silu)
    print("Linear Fwd")
    print(output_linear)
    print(output_ref_linear)
    print("SiLU Fwd")
    print(output_silu)
    print(output_ref_silu)
    print("Linear Bwd")
    output_ref_linear.backward(output_ref_linear)
    grad_input_linear = linear.backward(output_linear)
    grad_input_ref_linear = input_linear.grad
    print(grad_input_linear)
    print(grad_input_ref_linear)
    print("SiLU Bwd")
    output_ref_silu.backward(output_ref_silu)
    grad_input_silu = silu.backward(output_silu)
    grad_input_ref_silu = input_silu.grad
    print(grad_input_silu)
    print(grad_input_ref_silu)