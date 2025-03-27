import pytest
from src.get_augseqs_of_order import get_augseqs_of_order

def test_get_augseqs_of_order_order_1_len_2():
    """Test order 1, sequence length 2, binary alphabet."""
    augseqs = get_augseqs_of_order(2, 1, ['A', 'B'])
    expected = ['A*', 'B*', '*A', '*B']
    assert set(augseqs) == set(expected), f"Expected {expected}, got {augseqs}"

def test_get_augseqs_of_order_order_1_len_3():
    """Test order 1, sequence length 3, binary alphabet."""
    augseqs = get_augseqs_of_order(3, 1, ['A', 'B'])
    expected = ['A**', 'B**', '*A*', '*B*', '**A', '**B']
    assert set(augseqs) == set(expected), f"Expected {expected}, got {augseqs}"

def test_get_augseqs_of_order_order_2_len_3():
    """Test order 2, sequence length 3, binary alphabet."""
    augseqs = get_augseqs_of_order(3, 2, ['A', 'B'])
    expected = [
        'AB*', 'BA*', 'AA*', 'BB*',  # Positions 0,1
        'A*A', 'B*A', 'A*B', 'B*B',  # Positions 0,2
        '*AB', '*BB', '*AA', '*BA'   # Positions 1,2
    ]
    assert set(augseqs) == set(expected), f"Expected {expected}, got {augseqs}"

def test_get_augseqs_of_order_order_0():
    """Test order 0, sequence length 3 (all wildcards)."""
    augseqs = get_augseqs_of_order(3, 0, ['A', 'B'])
    expected = ['***']  # Just one sequence with all wildcards
    assert augseqs == expected, f"Expected {expected}, got {augseqs}"

def test_get_augseqs_of_order_count_check_ternary():
    """Count check for Order 2, sequence length 4, ternary alphabet."""
    # We should have C(4,2) combinations × 3² sequences per combination
    # = 6 × 9 = 54 sequences
    augseqs = get_augseqs_of_order(4, 2, ['A', 'B', 'C'])
    assert len(augseqs) == 6 * 9, f"Expected {6 * 9} sequences, got {len(augseqs)}"

def test_get_augseqs_of_order_count_check_binary():
    """Count check for Order 1, sequence length 5, binary alphabet."""
    # We should have C(5,1) combinations × 2¹ sequences per combination
    # = 5 × 2 = 10 sequences
    augseqs = get_augseqs_of_order(5, 1, ['A', 'B'])
    assert len(augseqs) == 5 * 2, f"Expected {5 * 2} sequences, got {len(augseqs)}"

def test_get_augseqs_of_order_validation():
    """Test validation logic."""
    # Test order > sequence length
    with pytest.raises(ValueError, match="Order .* cannot exceed sequence length"):
        get_augseqs_of_order(5, 6, ['A', 'B'])
    
    # Test negative order
    with pytest.raises(ValueError, match="Order must be non-negative"):
        get_augseqs_of_order(5, -1, ['A', 'B'])
    
    # Test empty alphabet
    with pytest.raises(ValueError, match="Alphabet must not be empty"):
        get_augseqs_of_order(5, 2, [])
    
    # Test wildcard in alphabet
    with pytest.raises(ValueError, match="Alphabet should not include the wildcard character"):
        get_augseqs_of_order(5, 2, ['A', 'B', '*']) 