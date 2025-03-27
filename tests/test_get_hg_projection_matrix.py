import pytest
import pandas as pd
import numpy as np
from scipy import sparse

from src.get_hg_projection_matrix import get_hg_projection_matrix
from src.get_augseqs_upto_order import get_augseqs_upto_order
def test_get_hg_projection_matrix_df():
    """Test DataFrame output format."""
    L = 3
    alphabet = ['A', 'C', 'G', 'T']
    pi_df = pd.DataFrame(index=range(L), columns=alphabet, data=1/len(alphabet))
    augseqs = get_augseqs_upto_order(alphabet=alphabet, seq_length=3, max_order=1)
    
    P = get_hg_projection_matrix(
        augseqs=augseqs, 
        alphabet=alphabet, 
        bg_type='custom', 
        bg_df=pi_df, 
        out_type='df'
    )
    
    # Check output type
    assert isinstance(P, pd.DataFrame)
    # Check dimensions
    assert P.shape == (len(augseqs), len(augseqs))
    # Check index and columns
    assert list(P.index) == augseqs
    assert list(P.columns) == augseqs

def test_get_hg_projection_matrix_sparse():
    """Test sparse matrix output format."""
    L = 3
    alphabet = ['A', 'C', 'G', 'T']
    pi_df = pd.DataFrame(index=range(L), columns=alphabet, data=1/len(alphabet))
    augseqs = get_augseqs_upto_order(alphabet=alphabet, seq_length=3, max_order=1)
    
    P = get_hg_projection_matrix(
        augseqs=augseqs, 
        alphabet=alphabet, 
        bg_type='custom', 
        bg_df=pi_df, 
        out_type='sparse'
    )
    
    # Check output type
    assert isinstance(P, sparse.csr_matrix)
    # Check dimensions
    assert P.shape == (len(augseqs), len(augseqs))

def test_get_hg_projection_matrix_invalid_output():
    """Test invalid output type."""
    L = 3
    alphabet = ['A', 'C', 'G', 'T']
    pi_df = pd.DataFrame(index=range(L), columns=alphabet, data=1/len(alphabet))
    augseqs = get_augseqs_upto_order(alphabet=alphabet, seq_length=3, max_order=1)
    
    with pytest.raises(ValueError, match="Invalid output type"):
        get_hg_projection_matrix(
            augseqs=augseqs, 
            alphabet=alphabet, 
            bg_type='custom', 
            bg_df=pi_df, 
            out_type='invalid'
        )

def test_get_hg_projection_matrix_single_position():
    """Test with single position sequences."""
    L = 1
    alphabet = ['A', 'B']
    pi_df = pd.DataFrame(index=range(L), columns=alphabet, data=1/len(alphabet))
    augseqs = ['A', 'B', '*']
    
    P = get_hg_projection_matrix(
        augseqs=augseqs, 
        alphabet=alphabet, 
        bg_type='custom', 
        bg_df=pi_df, 
        out_type='df'
    )
    
    # Check that matrix has correct structure
    assert P.shape == (3, 3)
    # Check that diagonal elements are correct
    assert np.isclose(P.loc['A', 'A'], 0.5)  # 1 - p_l^A
    assert np.isclose(P.loc['B', 'B'], 0.5)  # 1 - p_l^B
    assert np.isclose(P.loc['*', '*'], 1.0)  # eta = 1

def test_get_hg_projection_matrix_custom_wildcard():
    """Test with custom wildcard character."""
    L = 2
    alphabet = ['A', 'B']
    pi_df = pd.DataFrame(index=range(L), columns=alphabet, data=1/len(alphabet))
    augseqs = ['??','A?', 'B?', '?A', '?B']
    
    P = get_hg_projection_matrix(
        augseqs=augseqs, 
        alphabet=alphabet, 
        bg_type='custom', 
        bg_df=pi_df, 
        wildcard_char='?', 
        out_type='df'
    )
    
    # Check dimensions
    assert P.shape == (len(augseqs), len(augseqs))
    # Check that wildcard positions are handled correctly
    assert np.isclose(P.loc['??', '??'], 1.0)  # eta = 1 

def test_get_hg_projection_matrix_uniform_bg():
    """Test with uniform background distribution."""
    L = 2
    alphabet = ['A', 'C']
    augseqs = ['A*', 'C*', '*A', '*C', '**']
    
    # Use uniform background type
    P = get_hg_projection_matrix(
        augseqs=augseqs, 
        alphabet=alphabet, 
        bg_type='uniform',
        out_type='df'
    )
    
    # Check dimensions
    assert P.shape == (len(augseqs), len(augseqs))
    
    # For uniform background, pi_t = 1/len(alphabet) = 0.5 for each character
    # For the sequence '**', all suborbit sequences should map to it with value=1
    assert np.isclose(P.loc['**', '**'], 1.0)
    
    # For 'A*', expect specific values based on the uniform background model
    assert np.isclose(P.loc['A*', 'A*'], 0.5)  # 1 - pi_A = 1 - 0.5 = 0.5
    
    # Verify idempotency property: P @ P = P
    P_squared = P @ P
    pd.testing.assert_frame_equal(P, P_squared, check_exact=False, rtol=1e-10)

def test_get_hg_projection_matrix_wildtype_bg():
    """Test with wildtype background distribution."""
    L = 2
    alphabet = ['A', 'C']
    augseqs = ['A*', 'C*', '*A', '*C', '**']
    wt_seq = 'AC'  # Define wildtype sequence
    
    # Use wildtype background type
    P = get_hg_projection_matrix(
        augseqs=augseqs, 
        alphabet=alphabet, 
        bg_type='wildtype',
        wt_seq=wt_seq,
        out_type='df'
    )
    
    # Check dimensions
    assert P.shape == (len(augseqs), len(augseqs))
    
    # For wildtype background, pi_t = 1 if t == wt, otherwise 0
    # For the sequence '**', all suborbit sequences should map to it with value=1
    assert np.isclose(P.loc['**', '**'], 1.0)
    
    # For 'A*', the pi_A = 1 because 'A' matches the wildtype at position 0
    assert np.isclose(P.loc['A*', 'A*'], 0.0)  # 1 - pi_A = 1 - 1 = 0
    # For 'C*', the pi_C = 0 because 'C' doesn't match the wildtype at position 0
    assert np.isclose(P.loc['C*', 'C*'], 1.0)  # 1 - pi_C = 1 - 0 = 1
    
    # Verify idempotency property: P @ P = P
    P_squared = P @ P
    pd.testing.assert_frame_equal(P, P_squared, check_exact=False, rtol=1e-10)

def test_get_hg_projection_matrix_custom_bg():
    """Test with custom background distribution."""
    L = 2
    alphabet = ['A', 'C']
    augseqs = ['A*', 'C*', '*A', '*C', '**']
    
    # Create custom background distribution
    bg_df = pd.DataFrame(index=range(L), columns=alphabet)
    # Position 0: A=0.7, C=0.3
    bg_df.loc[0, 'A'] = 0.7
    bg_df.loc[0, 'C'] = 0.3
    # Position 1: A=0.4, C=0.6
    bg_df.loc[1, 'A'] = 0.4
    bg_df.loc[1, 'C'] = 0.6
    
    # Use custom background type
    P = get_hg_projection_matrix(
        augseqs=augseqs, 
        alphabet=alphabet, 
        bg_type='custom',
        bg_df=bg_df,
        out_type='df'
    )
    
    # Check dimensions
    assert P.shape == (len(augseqs), len(augseqs))
    
    # For custom background, pi_t is defined in bg_df
    # For the sequence '**', all suborbit sequences should map to it with value=1
    assert np.isclose(P.loc['**', '**'], 1.0)
    
    # For 'A*', the pi_A = 0.7 at position 0
    assert np.isclose(P.loc['A*', 'A*'], 0.3)  # 1 - pi_A = 1 - 0.7 = 0.3
    # For 'C*', the pi_C = 0.3 at position 0
    assert np.isclose(P.loc['C*', 'C*'], 0.7)  # 1 - pi_C = 1 - 0.3 = 0.7
    
    # Verify idempotency property: P @ P = P
    P_squared = P @ P
    pd.testing.assert_frame_equal(P, P_squared, check_exact=False, rtol=1e-10)

def test_invalid_bg_type_combinations():
    """Test invalid combinations of background type parameters."""
    L = 2
    alphabet = ['A', 'C']
    augseqs = ['A*', 'C*', '*A', '*C', '**']
    wt_seq = 'AC'
    bg_df = pd.DataFrame(index=range(L), columns=alphabet, data=0.5)
    
    # Test: wildtype bg_type without wt_seq
    with pytest.raises(ValueError, match="wt_seq must be provided if bg_type is 'wildtype'"):
        get_hg_projection_matrix(
            augseqs=augseqs, 
            alphabet=alphabet, 
            bg_type='wildtype',
            out_type='df'
        )
    
    # Test: custom bg_type without bg_df
    with pytest.raises(ValueError, match="bg_df must be provided if bg_type is 'custom'"):
        get_hg_projection_matrix(
            augseqs=augseqs, 
            alphabet=alphabet, 
            bg_type='custom',
            out_type='df'
        )
    
    # Test: uniform bg_type with wt_seq
    with pytest.raises(ValueError, match="wt_seq is not used and must be None if bg_type is not 'wildtype'"):
        get_hg_projection_matrix(
            augseqs=augseqs, 
            alphabet=alphabet, 
            bg_type='uniform',
            wt_seq=wt_seq,
            out_type='df'
        )
    
    # Test: uniform bg_type with bg_df
    with pytest.raises(ValueError, match="bg_df is not used and must be None if bg_type is not 'custom'"):
        get_hg_projection_matrix(
            augseqs=augseqs, 
            alphabet=alphabet, 
            bg_type='uniform',
            bg_df=bg_df,
            out_type='df'
        ) 