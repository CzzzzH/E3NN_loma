@simd
def add_(x : In[Array[float]],
         y : In[Array[float]],
         z : Out[Array[float]]):
    idx : int = thread_id()
    z[idx] = x[idx] + y[idx]

grad_add_ = rev_diff(add_)

@simd
def sub_(x : In[Array[float]],
         y : In[Array[float]],
         z : Out[Array[float]]):
    idx : int = thread_id()
    z[idx] = x[idx] - y[idx]

grad_sub_ = rev_diff(sub_)

@simd
def multiply_(x : In[Array[float]],
              y : In[Array[float]],
              z : Out[Array[float]]):
    idx : int = thread_id() 
    z[idx] = x[idx] * y[idx]

grad_multiply_ = rev_diff(multiply_)

@simd
def divide_(x : In[Array[float]],
            y : In[Array[float]],
            z : Out[Array[float]]):
    idx : int = thread_id()
    z[idx] = x[idx] / y[idx]

grad_divide_ = rev_diff(divide_)

@simd
def sqrt_(input: In[Array[float]],
          output: Out[Array[float]]):
    idx : int = thread_id()
    output[idx] = sqrt(input[idx])

grad_sqrt_ = rev_diff(sqrt_)

@simd
def sum_(input: In[Array[float]],
         output: Out[Array[float]],
         in_features: In[Array[int]]):
    batch_idx : int = thread_id() / in_features[0]
    feature_idx : int = thread_id() - batch_idx * in_features[0]
    atomic_add(output[batch_idx], input[batch_idx * in_features[0] + feature_idx])

grad_sum_ = rev_diff(sum_)

@simd
def mean_(input: In[Array[float]],
          output: Out[Array[float]],
          in_features: In[Array[int]]):
    batch_idx : int = thread_id() / in_features[0]
    feature_idx : int = thread_id() - batch_idx * in_features[0]
    atomic_add(output[batch_idx], input[batch_idx * in_features[0] + feature_idx] / int2float(in_features[0]))

grad_mean_ = rev_diff(mean_)

@simd
def linear_(input: In[Array[float]],
            weight: In[Array[float]],
            bias: In[Array[float]],
            output: Out[Array[float]],
            in_features: In[Array[int]],
            out_features: In[Array[int]]):
    batch_idx : int = thread_id() / out_features[0]
    feature_idx : int = thread_id() - batch_idx * out_features[0]
    i : int = 0
    tmp : float = 0.0
    while (i < in_features[0], max_iter := 1000):
        tmp = tmp + input[batch_idx * in_features[0] + i] * weight[feature_idx * in_features[0] + i]
        i = i + 1
    tmp = tmp + bias[feature_idx]
    output[batch_idx * out_features[0] + feature_idx] = tmp

grad_linear_ = rev_diff(linear_)

@simd
def relu_(input: In[Array[float]],
          output: Out[Array[float]]):
    idx : int = thread_id()
    if input[idx] < 0.0:
        output[idx] = 0.0
    else:
        output[idx] = input[idx]

grad_relu_ = rev_diff(relu_)

@simd
def silu_(input: In[Array[float]],
          output: Out[Array[float]]):
    idx : int = thread_id()
    output[idx] = input[idx] / (1.0 + exp(-input[idx]))
        
grad_silu_ = rev_diff(silu_)

@simd
def sigmoid_(input: In[Array[float]],
             output: Out[Array[float]]):
    idx : int = thread_id()
    output[idx] = 1.0 / (1.0 + exp(-input[idx]))

grad_sigmoid_ = rev_diff(sigmoid_)

@simd
def mse_(x : In[Array[float]],
         y : In[Array[float]],
         z : Out[Array[float]],
         batch_size : In[Array[int]],
         out_features : In[Array[int]]):
    batch_idx : int = thread_id() / out_features[0]
    feature_idx : int = thread_id() - batch_idx * out_features[0]
    N : float = int2float(batch_size[0] * out_features[0])
    atomic_add(z[0], (x[batch_idx * out_features[0] + feature_idx] - y[batch_idx * out_features[0] + feature_idx]) * \
        (x[batch_idx * out_features[0] + feature_idx] - y[batch_idx * out_features[0] + feature_idx]) / N)

grad_mse_ = rev_diff(mse_)
