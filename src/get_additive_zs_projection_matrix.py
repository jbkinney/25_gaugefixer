import numpy as np
from scipy.sparse import lil_matrix

def get_additive_zs_projection_matrix(L, alphabet):
    """
    Generate a projection matrix for an additive model in Hilbert space zero-sum gauge.
    
    Parameters:
    -----------
    L : int
        Length of sequences.
    alphabet : list
        List of characters in the alphabet.
        
    Returns:
    --------
    mat : scipy.sparse.csr_matrix
        Sparse matrix for the projection operation.
    """
    alpha = len(alphabet)
    
    # Get matrix size
    N = 1 + alpha*L
    
    # Initialize matrix
    mat = lil_matrix((N, N))
    
    # Set first row
    mat[0,0] = 1.0  
    mat[0,1:] = 1/alpha
    
    # Set block diagonal elements
    shift = 1
    block = np.eye(alpha) - 1/alpha
    for i in range(L):
        start = shift + alpha*i
        stop = start + alpha
        mat[start:stop, start:stop] = block
        
    # Convert to CSR format for efficient operations
    mat = mat.tocsr()
    return mat 