@simd
def vector_add(x : In[Array[float]],
               y : In[Array[float]],
               z : Out[Array[float]]):
    i : int = thread_id()
    z[i] = x[i] + y[i]

grad_vector_add = rev_diff(vector_add)

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
    while (i < in_features, max_iter := 1000):
        output[batch_idx * out_features + feature_idx] = output[batch_idx * out_features + feature_idx] + \
            input[batch_idx * in_features + i] * weight[feature_idx * in_features + i]
        i = i + 1
    output[batch_idx * out_features + feature_idx] = output[batch_idx * out_features + feature_idx] + bias[feature_idx]

grad_linear = rev_diff(linear)

@simd
def silu(input: In[Array[float]],
         output: Out[Array[float]],
         out_features: In[int]):
    batch_idx : int = thread_id() / out_features
    feature_idx : int = thread_id() - batch_idx * out_features
    output[batch_idx * out_features + feature_idx] = input[batch_idx * out_features + feature_idx] * \
        1.0 / (1.0 + exp(-input[batch_idx * out_features + feature_idx]))

grad_silu = rev_diff(silu)