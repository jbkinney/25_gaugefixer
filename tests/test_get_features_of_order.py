import pytest
from gaugefixer.features.get_features_of_order import get_features_of_order
from gaugefixer.features.petti_feature import PettiFeature

def test_get_features_of_order_basic():
    """Test basic functionality with simple inputs."""
    # Order 1, sequence length 2, binary alphabet
    features = get_features_of_order(L=2, order=1, alphabet=['A', 'B'])
    
    # Expected: sorted list of all combinations of positions and characters
    expected = [
        ((0,), 'A'),  # Position 0, character A
        ((0,), 'B'),  # Position 0, character B
        ((1,), 'A'),  # Position 1, character A
        ((1,), 'B'),  # Position 1, character B
    ]
    
    assert features == expected
    assert len(features) == 4  # 2 positions * 2 characters

def test_get_features_of_order_zero():
    """Test with order 0, which should return a special case."""
    features = get_features_of_order(L=3, order=0, alphabet=['A', 'B', 'C'])
    
    # Order 0 means empty tuple of positions and empty string
    expected = [((), '')]
    
    assert features == expected
    assert len(features) == 1

def test_get_features_of_order_alphabet_size():
    """Test the effect of alphabet size on the number of features."""
    # 2 positions, order 2, alphabet size 2
    features1 = get_features_of_order(L=2, order=2, alphabet=['0', '1'])
    
    # 2 positions, order 2, alphabet size 3
    features2 = get_features_of_order(L=2, order=2, alphabet=['0', '1', '2'])
    
    # For order 2, we expect alphabet_size^2 different possible subsequences
    assert len(features1) == 4  # 2^2 = 4
    assert len(features2) == 9  # 3^2 = 9

def test_get_features_of_order_complex():
    """Test with more complex inputs."""
    # Order 2, sequence length 3, binary alphabet
    features = get_features_of_order(L=3, order=2, alphabet=['A', 'B'])
    
    # Expected: all combinations of 2 positions from 3 positions (3 combinations)
    # and all combinations of 2 characters from alphabet (4 combinations)
    # Total: 3 * 4 = 12 features
    expected_count = 12
    
    # Check a few specific features
    assert ((0, 1), 'AA') in features
    assert ((0, 2), 'BB') in features
    assert ((1, 2), 'AB') in features
    
    assert len(features) == expected_count

def test_get_features_of_order_validation():
    """Test input validation."""
    # Test with invalid inputs
    with pytest.raises(AssertionError):
        get_features_of_order(L=-1, order=1, alphabet=['A', 'B'])
    
    with pytest.raises(AssertionError):
        get_features_of_order(L=5, order=-1, alphabet=['A', 'B'])
    
    with pytest.raises(AssertionError):
        get_features_of_order(L=2, order=3, alphabet=['A', 'B'])
    
    with pytest.raises(AssertionError):
        get_features_of_order(L=5, order=2, alphabet=[])
    
    # Test type checking (note: these might be caught by typeguard)
    with pytest.raises(Exception):  # Either AssertionError or TypeCheckError
        get_features_of_order(L=5, order=2, alphabet=[1, 2, 3])

def test_get_features_of_order_sorting():
    """Test that features are properly sorted."""
    features = get_features_of_order(L=3, order=2, alphabet=['B', 'A'])
    
    # Check that positions are sorted first by length, then by position values
    # then by subsequence string
    for i in range(len(features) - 1):
        current = features[i]
        next_feature = features[i + 1]
        
        # Either the current position tuple is "less than" the next
        # or they're equal and the current subsequence is "less than" the next
        assert (
            current[0] < next_feature[0] or 
            (current[0] == next_feature[0] and current[1] <= next_feature[1])
        ) 