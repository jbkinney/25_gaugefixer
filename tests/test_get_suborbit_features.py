import pytest
from src._get_suborbit_features import _get_suborbit_features

def test_get_suborbit_features_single_position():
    """Test with a single position, DNA alphabet."""
    features = _get_suborbit_features('*A**', ['A', 'C', 'G', 'T'])
    expected = ['****', '*A**', '*C**', '*G**', '*T**']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_suborbit_features_two_positions():
    """Test with two positions, binary alphabet."""
    features = _get_suborbit_features('A*B', ['A', 'B'])
    # Should be:['***', '**A', '**B', 'A**', 'A*A', 'A*B', 'B**', 'B*A', 'B*B']
    assert len(features) == 9, f"Expected 9 sequences, got {len(features)}"
    assert features == ['***', '**A', '**B', 'A**', 'A*A', 'A*B', 'B**', 'B*A', 'B*B']


def test_get_suborbit_features_custom_wildcard():
    """Test with a custom wildcard character."""
    features = _get_suborbit_features('?A??', ['A', 'B'], wildcard_char='?')
    expected = ['????', '?A??', '?B??']
    assert set(features) == set(expected), f"Expected {expected}, got {features}"

def test_get_suborbit_features_empty_alphabet():
    """Test with empty alphabet."""
    with pytest.raises(AssertionError, match="Alphabet must not be empty"):
        _get_suborbit_features('*A**', [])

def test_get_suborbit_features_wildcard_in_alphabet():
    """Test with wildcard in alphabet."""
    with pytest.raises(AssertionError, match="Alphabet should not include the wildcard character"):
        _get_suborbit_features('*A**', ['A', 'B', '*']) 