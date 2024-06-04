@simd
def parallel_sqrt(input: In[Array[float]],
          output: Out[Array[float]]):
    idx : int = thread_id()
    output[idx] = sqrt(input[idx])

grad_sqrt_ = rev_diff(parallel_sqrt)
