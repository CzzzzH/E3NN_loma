class Matrix:
    data: Array[float]
    num_rows: int
    num_cols: int
    
class Vector:
    data: Array[float]
    size: int

@simd
def vector_add(x : In[Array[float]],
               y : In[Array[float]],
               z : Out[Array[float]]):
    i : int = thread_id()
    z[i] = x[i] + y[i]