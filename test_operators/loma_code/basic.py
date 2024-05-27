class Matrix:
    data: Array[float]
    num_rows: int
    num_cols: int
    
class Vector:
    data: Array[float]
    size: int
    
# Can not work!
# @simd
# def vector_add(x : In[Vector],
#                y : In[Vector],
#                z : Out[Vector]):
#     i : int = thread_id()
#     # z.data[i] = x.data[i] + y.data[i]

@simd
def vector_add(x : In[Array[float]],
               y : In[Array[float]],
               z : Out[Array[float]]):
    i : int = thread_id()
    z[i] = x[i] + y[i]

@simd
def matrix_mul(x : In[Array[float]],
               y : In[Array[float]],
               z : Out[Array[float]],
               z_row: In[int],
               mid: In[int],
               z_col: In[int]):
    k : int = thread_id()
    i : int = 0
    j : int = 0
    while (i < z_row, max_iter := 100):
        j = 0
        while (j < z_col, max_iter := 100):
            atomic_add(z[z_col * i + j], x[mid * i + k] * y[z_col * k + j])
            j = j + 1
        i = i + 1
    