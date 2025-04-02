import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.features.get_features_upto_order import get_features_upto_order
from src.features.get_features_of_order import get_features_of_order

def test_get_features_upto_order_order_1_len_2():
    """Test up to order 1, sequence length 2, binary alphabet."""
    features = get_features_upto_order(2, 1, ['A', 'B'])
    expected = ['**', 'A*', 'B*', '*A', '*B']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_features_upto_order_order_1_len_3():
    """Test up to order 1, sequence length 3, binary alphabet."""
    features = get_features_upto_order(3, 1, ['A', 'B'])
    expected = ['***', 'A**', 'B**', '*A*', '*B*', '**A', '**B']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_features_upto_order_order_2_len_3():
    """Test up to order 2, sequence length 3, binary alphabet."""
    features = get_features_upto_order(3, 2, ['A', 'B'])
    # Order 0: ['***']
    # Order 1: ['A**', 'B**', '*A*', '*B*', '**A', '**B']
    # Order 2: ['AB*', 'BA*', 'AA*', 'BB*', 'A*A', 'B*A', 'A*B', 'B*B', '*AB', '*BB', '*AA', '*BA']
    # Total: 1 + 6 + 12 = 19 sequences
    assert len(features) == 19, f"Expected 19 sequences, got {len(features)}"

def test_get_features_upto_order_by_order_sum():
    """Verify results against individual calls to get_features_of_order."""
    # Test for sequence length 3, binary alphabet, up to order 2
    upto_seqs = get_features_upto_order(3, 2, ['A', 'B'])
    
    # Get individual order sequences
    order0 = get_features_of_order(3, 0, ['A', 'B'])
    order1 = get_features_of_order(3, 1, ['A', 'B'])
    order2 = get_features_of_order(3, 2, ['A', 'B'])
    
    # Verify combined count matches
    assert len(upto_seqs) == len(order0) + len(order1) + len(order2), "Count mismatch"
    
    # Verify combined contents match
    combined = order0 + order1 + order2
    assert set(upto_seqs) == set(combined), "Content mismatch"

def test_get_features_upto_order_order_0():
    """Test up to order 0, sequence length 3 (only all wildcards)."""
    features = get_features_upto_order(3, 0, ['A', 'B'])
    expected = ['***']  # Just one sequence with all wildcards
    assert features == expected, f"Expected {expected}, got {features}"

def test_get_features_upto_order_count_check_ternary():
    """Count check for up to Order 2, sequence length 4, ternary alphabet."""
    # Order 0: 1 sequence
    # Order 1: 4 positions * 3 characters = 12 sequences
    # Order 2: C(4,2) combinations * 3² = 6 * 9 = 54 sequences
    # Total: 1 + 12 + 54 = 67 sequences
    features = get_features_upto_order(4, 2, ['A', 'B', 'C'])
    assert len(features) == 67, f"Expected 67 sequences, got {len(features)}"

def test_get_features_upto_order_max_order_equals_seq_length():
    """Count check for maximum Order = sequence length."""
    # For length 3, binary alphabet, all possible augmented sequences
    features = get_features_upto_order(3, 3, ['A', 'B'])
    # Order 0: 1 sequence
    # Order 1: 3 positions * 2 characters = 6 sequences
    # Order 2: C(3,2) combinations * 2² = 3 * 4 = 12 sequences
    # Order 3: C(3,3) combinations * 2³ = 1 * 8 = 8 sequences
    # Total: 1 + 6 + 12 + 8 = 27 sequences
    assert len(features) == 27, f"Expected 27 sequences, got {len(features)}"

def test_get_features_upto_order_validation():
    """Test validation logic."""
    # Test max order > sequence length
    with pytest.raises(ValueError, match="Maximum order .* cannot exceed sequence length"):
        get_features_upto_order(5, 6, ['A', 'B'])
    
    # Test negative max order
    with pytest.raises(ValueError, match="Maximum order must be non-negative"):
        get_features_upto_order(5, -1, ['A', 'B']) 