import pytest
from src.get_suborbit_augseqs import get_suborbit_augseqs

def test_get_suborbit_augseqs_single_position():
    """Test with a single position, DNA alphabet."""
    augseqs = get_suborbit_augseqs('*A**', ['A', 'C', 'G', 'T'])
    expected = ['****', '*A**', '*C**', '*G**', '*T**']
    assert set(augseqs) == set(expected), f"Expected {expected}, got {augseqs}"

def test_get_suborbit_augseqs_two_positions():
    """Test with two positions, binary alphabet."""
    augseqs = get_suborbit_augseqs('A*B', ['A', 'B'])
    # Should be:['***', '**A', '**B', 'A**', 'A*A', 'A*B', 'B**', 'B*A', 'B*B']
    assert len(augseqs) == 9, f"Expected 9 sequences, got {len(augseqs)}"
    assert augseqs == ['***', '**A', '**B', 'A**', 'A*A', 'A*B', 'B**', 'B*A', 'B*B']


def test_get_suborbit_augseqs_custom_wildcard():
    """Test with a custom wildcard character."""
    augseqs = get_suborbit_augseqs('?A??', ['A', 'B'], wildcard_char='?')
    expected = ['????', '?A??', '?B??']
    assert set(augseqs) == set(expected), f"Expected {expected}, got {augseqs}"

def test_get_suborbit_augseqs_empty_alphabet():
    """Test with empty alphabet."""
    with pytest.raises(AssertionError, match="Alphabet must not be empty"):
        get_suborbit_augseqs('*A**', [])

def test_get_suborbit_augseqs_wildcard_in_alphabet():
    """Test with wildcard in alphabet."""
    with pytest.raises(AssertionError, match="Alphabet should not include the wildcard character"):
        get_suborbit_augseqs('*A**', ['A', 'B', '*']) 