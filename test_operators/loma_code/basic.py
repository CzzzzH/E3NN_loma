@simd
def add(x : In[Array[float]],
        y : In[Array[float]],
        z : Out[Array[float]],
        out_features : In[int]):
    batch_idx : int = thread_id() / out_features
    feature_idx : int = thread_id() - batch_idx * out_features
    z[batch_idx * out_features + feature_idx] = \
        x[batch_idx * out_features + feature_idx] + \
        y[batch_idx * out_features + feature_idx]

grad_add = rev_diff(add)

@simd
def sum(input: In[Array[float]],
        output: Out[Array[float]],
        in_features: In[int]):
    batch_idx : int = thread_id() / in_features
    feature_idx : int = thread_id() - batch_idx * in_features
    atomic_add(output[batch_idx], input[batch_idx * in_features + feature_idx])

grad_sum = rev_diff(sum)

@simd
def linear(input: In[Array[float]],
           weight: In[Array[float]],
           bias: In[Array[float]],
           output: Out[Array[float]],
           in_features: In[int],
           out_features: In[int]):
    batch_idx : int = thread_id() / out_features
    feature_idx : int = thread_id() - batch_idx * out_features
    i : int = 0
    tmp : float = 0.0
    while (i < in_features, max_iter := 1000):
        tmp = tmp + input[batch_idx * in_features + i] * weight[feature_idx * in_features + i]
        i = i + 1
    tmp = tmp + bias[feature_idx]
    output[batch_idx * out_features + feature_idx] = tmp

grad_linear = rev_diff(linear)

@simd
def relu(input: In[Array[float]],
         output: Out[Array[float]],
         out_features: In[int]):
    batch_idx : int = thread_id() / out_features
    feature_idx : int = thread_id() - batch_idx * out_features
    if input[batch_idx * out_features + feature_idx] < 0:
        output[batch_idx * out_features + feature_idx] = 0
    else:
        output[batch_idx * out_features + feature_idx] = input[batch_idx * out_features + feature_idx]

grad_relu = rev_diff(relu)

@simd
def silu(input: In[Array[float]],
         output: Out[Array[float]],
         out_features: In[int]):
    batch_idx : int = thread_id() / out_features
    feature_idx : int = thread_id() - batch_idx * out_features
    output[batch_idx * out_features + feature_idx] = input[batch_idx * out_features + feature_idx] * \
        1.0 / (1.0 + exp(-input[batch_idx * out_features + feature_idx]))

grad_silu = rev_diff(silu)

