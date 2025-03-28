import math
import numpy as np
from scipy.sparse import lil_matrix

def _elongate_block_horizontally(block, factor):
    """
    Horizontally elongates a 2D numpy array by duplicating each column multiple times.
    
    Args:
        block: 2D numpy array to elongate
        factor: Number of times each column should be duplicated
        
    Returns:
        A new 2D numpy array with the same number of rows as block, but with
        each column duplicated factor times
    """
    num_cols = block.shape[1]
    num_rows = block.shape[0]
    columns = []
    for c in range(num_cols):
        for i in range(factor):
            columns.append(block[:,c].reshape(num_rows,1))
    return np.concatenate(columns, axis=1)


def get_pairwise_zs_projection_matrix(L, alphabet):
    """
    Generates a sparse projection matrix for pairwise zero-sum gauge fixing.
    
    This function constructs a specific structured sparse matrix used in 
    zero-sum gauge-fixing calculations for sequence analysis. The matrix structure
    contains several specialized blocks that enable efficient computation
    of projections into the zero-sum gauge.
    
    Args:
        L: Integer representing the sequence length
        alphabet: Collection (e.g., list or string) of characters in the alphabet
                 (e.g., 'ACGT' for DNA sequences)
    
    Returns:
        A scipy.sparse.csr_matrix of shape (N, N) where 
        N = 1 + (alpha*L) + (alpha^2*comb(L,2)) and alpha is the alphabet size.
        This matrix represents the projection operation for the zero-sum gauge.
    """
    
    alpha = len(alphabet)
    alpha_inv = 1/alpha
    p0 = 0
    p1 = p0 + 1
    p2 = p1 + alpha*L
    N = p2 + alpha**2*math.comb(L, 2)
    mat = lil_matrix((N, N))

    # First row
    mat[p0:p1,p0:p1] = 1.0
    mat[p0:p1,p1:p2] = alpha_inv
    mat[p0:p1,p2:N] = alpha_inv**2

    eye = np.eye(alpha)

    block1 = (1 - alpha_inv)*eye - alpha_inv*(1 - eye)
    for i in range(L):
        start = p1+i*alpha
        stop = start + alpha
        mat[start:stop,start:stop] = block1
        
    block2 = (1 - alpha_inv)*block1
    for i in range(math.comb(L, 2)*alpha):
        start = p2+i*alpha
        stop = start + alpha
        mat[start:stop,start:stop] = block2
        
    block3 = - alpha_inv*block1
    p = p2
    for k in range(1,L):
        for i in range(k*alpha):
            for j in range(k*alpha):
                if i != j:
                    if (i-j) % k == 0:
                        rstart = p+i*alpha
                        rstop = rstart + alpha
                        cstart = p+j*alpha
                        cstop = cstart + alpha
                        mat[rstart:rstop,cstart:cstop] = block3
        p += k*alpha**2
        
    block4 = (1-alpha_inv)*alpha_inv*eye - alpha_inv*alpha_inv*(1-eye)
    rp = p1
    cp = p2
    for k in range(1,L):
        # Add block 4
        for i in range(k):
            for j in range(alpha):
                rstart = rp+i*alpha
                rstop = rstart + alpha
                cstart = cp+(k*j+i)*alpha
                cstop = cstart + alpha
                if rstop <= N and cstop <= N:
                    mat[rstart:rstop,cstart:cstop] = block4
        
        # Add long block
        long_block = _elongate_block_horizontally(block4,factor=k*alpha)
        rstart = rp+k*alpha
        rstop = rstart + alpha
        cstart = cp
        cstop = cstart + k*alpha*alpha
        mat[rstart:rstop,cstart:cstop] = long_block
        
        cp += k*alpha**2
                
    mat = mat.tocsr()
    return mat
            