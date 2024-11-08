import numpy as np

def my_reshape(vec, n_rows):
    """
    Reshape vector into matrix form with interleaved columns
    """
    n_cols = len(vec)
    vec = vec.reshape(1, n_cols)
    mat = np.ones((n_rows, n_cols))
    
    for kk in range(n_rows):
        ll = n_rows - kk
        mat[ll-1, :-(ll-1) if ll > 1 else None] = vec[0, ll-1:]
    
    return mat
