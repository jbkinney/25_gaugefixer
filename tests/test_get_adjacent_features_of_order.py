import pytest
from purgatory.get_adjacent_features_of_order import get_adjacent_features_of_order

def test_get_adjacent_features_of_order_example():
    """Test the example from the docstring."""
    features = get_adjacent_features_of_order(seq_length=3, order=2, alphabet=['A', 'B'])
    expected = ['AA*', 'AB*', 'BA*', 'BB*', '*AA', '*AB', '*BA', '*BB']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_adjacent_features_of_order_seq_len_3_order_1():
    """Test with sequence length 3, order 1, binary alphabet."""
    features = get_adjacent_features_of_order(seq_length=3, order=1, alphabet=['A', 'B'])
    expected = ['A**', 'B**', '*A*', '*B*', '**A', '**B']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_adjacent_features_of_order_seq_len_4_order_2():
    """Test with sequence length 4, order 2, binary alphabet."""
    features = get_adjacent_features_of_order(seq_length=4, order=2, alphabet=['A', 'B'])
    expected = [
        'AA**', 'AB**', 'BA**', 'BB**',  # Positions 0,1
        '*AA*', '*AB*', '*BA*', '*BB*',  # Positions 1,2
        '**AA', '**AB', '**BA', '**BB'   # Positions 2,3
    ]
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_adjacent_features_of_order_seq_len_3_order_3():
    """Test with sequence length 3, order 3 (spans the entire sequence)."""
    features = get_adjacent_features_of_order(seq_length=3, order=3, alphabet=['A', 'B'])
    # With order=3 and seq_length=3, we should get all 8 possible sequences of length 3
    expected = ['AAA', 'AAB', 'ABA', 'ABB', 'BAA', 'BAB', 'BBA', 'BBB']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_adjacent_features_of_order_ternary_alphabet():
    """Test with a ternary alphabet."""
    features = get_adjacent_features_of_order(seq_length=3, order=2, alphabet=['A', 'B', 'C'])
    # With a ternary alphabet, we should have 3² = 9 features for each sliding window
    # With seq_length=3 and order=2, we have 2 sliding windows
    assert len(features) == 2 * 9, f"Expected {2 * 9} features, got {len(features)}"
    # Check for a few expected patterns
    assert 'AA*' in features
    assert 'BC*' in features
    assert '*CA' in features

def test_get_adjacent_features_of_order_count_check():
    """Count check for different sequence lengths and orders."""
    # For seq_length=5, order=2, binary alphabet:
    # We should have (5-2+1) = 4 sliding windows, each with 2² = 4 patterns
    features = get_adjacent_features_of_order(seq_length=5, order=2, alphabet=['A', 'B'])
    assert len(features) == 4 * 4, f"Expected {4 * 4} features, got {len(features)}"
    
    # For seq_length=6, order=3, binary alphabet:
    # We should have (6-3+1) = 4 sliding windows, each with 2³ = 8 patterns
    features = get_adjacent_features_of_order(seq_length=6, order=3, alphabet=['A', 'B'])
    assert len(features) == 4 * 8, f"Expected {4 * 8} features, got {len(features)}"

def test_get_adjacent_features_of_order_validation():
    """Test validation logic."""
    # Test order < 1
    with pytest.raises(ValueError, match="Order must be greater or equal to 1"):
        get_adjacent_features_of_order(seq_length=3, order=0, alphabet=['A', 'B'])
    
    # Test order > sequence length
    with pytest.raises(ValueError, match="Order .* cannot exceed sequence length"):
        get_adjacent_features_of_order(seq_length=3, order=4, alphabet=['A', 'B'])

def test_get_adjacent_features_of_order_sliding_window():
    """Test that the function correctly applies sliding windows."""
    # For seq_length=5, order=2, we should see the pattern sliding across the sequence
    features = get_adjacent_features_of_order(seq_length=5, order=2, alphabet=['A'])
    expected = ['AA***', '*AA**', '**AA*', '***AA']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_adjacent_features_of_order_edge_case_order_equals_seq_length():
    """Test the edge case where order equals sequence length."""
    features = get_adjacent_features_of_order(seq_length=2, order=2, alphabet=['A', 'B'])
    expected = ['AA', 'AB', 'BA', 'BB']
    assert set(features) == set(expected), f"Expected {expected}, got {features}" 