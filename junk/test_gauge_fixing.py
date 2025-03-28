import numpy as np
import pandas as pd
import pytest
from junk.gauge_fixing import gauge_fix_sequences

def test_gauge_fix_sequences():
    """Test the gauge_fix_sequences function with example data using augmented sequences."""
    # Create test data with augmented sequences (using '*' as wildcards)
    test_df = pd.DataFrame({
        'sequence': ['A*C', 'AB*', '*BC', 'A*C', 'AB*'],
        'theta': [0.1, 0.2, 0.3, 0.1, 0.2]
    })
    
    # Create probability matrix as DataFrame
    # Rows: sequence positions (0, 1, 2)
    # Columns: valid characters (A, B, C)
    test_p_lc = pd.DataFrame({
        'A': [0.7, 0.3, 0.0],
        'B': [0.2, 0.6, 0.1],
        'C': [0.0, 0.0, 0.8]
    }, index=[0, 1, 2])
    
    # Test with lambda = 1.0
    result_df = gauge_fix_sequences(test_df, lambda_param=1.0, p_lc=test_p_lc)
    
    # Basic assertions
    assert len(result_df) == len(test_df), "Output DataFrame should have same length as input"
    assert all(result_df['sequence'] == test_df['sequence']), "Sequences should remain unchanged"
    
    # Check that repeated sequences have same theta values
    seq1_thetas = result_df[result_df['sequence'] == 'A*C']['theta'].values
    seq2_thetas = result_df[result_df['sequence'] == 'AB*']['theta'].values
    assert np.allclose(seq1_thetas, seq1_thetas[0]), "All A*C thetas should be equal"
    assert np.allclose(seq2_thetas, seq2_thetas[0]), "All AB* thetas should be equal"

def test_gauge_fix_sequences_infinite_lambda():
    """Test gauge fixing with infinite lambda value."""
    test_df = pd.DataFrame({
        'sequence': ['A*C', 'AB*', '*BC'],
        'theta': [0.1, 0.2, 0.3]
    })
    
    test_p_lc = pd.DataFrame({
        'A': [0.7, 0.3, 0.0],
        'B': [0.2, 0.6, 0.1],
        'C': [0.0, 0.0, 0.8]
    }, index=[0, 1, 2])
    
    result_df_inf = gauge_fix_sequences(test_df, lambda_param='inf', p_lc=test_p_lc)
    assert len(result_df_inf) == len(test_df), "Output DataFrame should have same length as input"

def test_gauge_fix_sequences_trivial_gauge():
    """Test gauge fixing with lambda = 0 (trivial gauge)."""
    test_df = pd.DataFrame({
        'sequence': ['A*C', 'AB*', '*BC'],
        'theta': [0.1, 0.2, 0.3]
    })
    
    test_p_lc = pd.DataFrame({
        'A': [0.7, 0.3, 0.0],
        'B': [0.2, 0.6, 0.1],
        'C': [0.0, 0.0, 0.8]
    }, index=[0, 1, 2])
    
    result_df_trivial = gauge_fix_sequences(test_df, lambda_param=0.0, p_lc=test_p_lc)
    assert len(result_df_trivial) == len(test_df), "Output DataFrame should have same length as input"
    
    # Check if wildcards have zero values in trivial gauge
    wild_seqs = [seq for seq in test_df['sequence'].unique() if '*' in seq]
    for seq in wild_seqs:
        theta_val = result_df_trivial[result_df_trivial['sequence'] == seq]['theta'].values[0]
        assert np.isclose(theta_val, 0.0), f"Sequence {seq} with wildcard should have theta=0 in trivial gauge"

def test_gauge_fix_sequences_custom_columns():
    """Test gauge fixing with custom column names."""
    renamed_df = pd.DataFrame({
        'seq': ['A*C', 'AB*', '*BC'],
        'param': [0.1, 0.2, 0.3]
    })
    
    test_p_lc = pd.DataFrame({
        'A': [0.7, 0.3, 0.0],
        'B': [0.2, 0.6, 0.1],
        'C': [0.0, 0.0, 0.8]
    }, index=[0, 1, 2])
    
    result_df_renamed = gauge_fix_sequences(
        renamed_df, 
        lambda_param=1.0, 
        p_lc=test_p_lc,
        sequence_col='seq',
        theta_col='param'
    )
    assert len(result_df_renamed) == len(renamed_df), "Output DataFrame should have same length as input" 