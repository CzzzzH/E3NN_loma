class Matrix:
    data: Array[float]
    num_rows: int
    num_cols: int
    
class Vector:
    data: Array[float]
    size: int

@simd
def parallel_add(x : In[Array[float]],
               y : In[Array[float]],
               z : Out[Array[float]]):
    i : int = thread_id()
    z[i] = x[i] + y[i]

# def vector_add2(x : In[Array[float]],
#                y : In[Array[float]],
#                z : Out[Array[float]],
#                n : In[int]):
#     i : int = 0
#     while (i < n, max_iter := 1000):
#         z[i] = x[i] + y[i]
#         i = i + 1

def vector_add(x : In[Vector],
               y : In[Vector],
               z : Out[Vector]):
    # i : int = 0
    # z.size = x.size
    # while (i < z.size, max_iter := 1000):
    #     z.data[i] = x.data[i] + y.data[i]
    #     i = i + 1
    # vector_add2(x.data, y.data, z.data, x.size)
    z.size = x.size
    parallel_add(x.data, y.data, z.data, z.size)
    