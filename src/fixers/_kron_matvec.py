import numpy as np

def _kron_matvec(matrices, vector):
    """
    Efficiently compute the matrix-vector product where the matrix is a Kronecker product of smaller matrices.
    
    Parameters:
    -----------
    matrices: list of np.ndarray
        List of matrices whose Kronecker product forms the large matrix
    vector: np.ndarray
        Vector to multiply with the Kronecker product matrix
        
    Returns:
    --------
    np.ndarray
        Result of the matrix-vector multiplication
    """
    # Get dimensions of each matrix
    dimensions = [m.shape[0] for m in matrices]
    
    # Calculate total size to verify compatibility
    total_size = np.prod(dimensions)
    
    if len(vector) != total_size:
        raise ValueError(f"Vector length ({len(vector)}) does not match the Kronecker product matrix size ({total_size})")
    
    # Reshape the vector to a multidimensional array
    x_reshaped = vector.reshape(dimensions)
    
    # Apply each matrix in reverse order
    for i in range(len(matrices) - 1, -1, -1):
        # Transpose x_reshaped to bring the current dimension to the front
        perm = np.concatenate([[i], np.arange(i), np.arange(i + 1, len(matrices))])
        x_transposed = np.transpose(x_reshaped, perm)
        
        # Reshape for matrix multiplication
        x_mat_shape = (dimensions[i], -1)
        x_mat = x_transposed.reshape(x_mat_shape)
        
        # Apply the current matrix
        result = matrices[i] @ x_mat
        
        # Reshape back
        new_dim = list(x_transposed.shape)
        new_dim[0] = matrices[i].shape[0]
        x_reshaped = result.reshape(new_dim)
        
        # Transpose back to original dimension order
        inv_perm = np.zeros(len(matrices), dtype=int)
        inv_perm[perm] = np.arange(len(matrices))
        x_reshaped = np.transpose(x_reshaped, inv_perm)
    
    # Flatten to get the result vector
    return x_reshaped.flatten()
