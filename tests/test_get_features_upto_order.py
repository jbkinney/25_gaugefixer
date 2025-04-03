import pytest
from gaugefixer.features.get_features_upto_order import get_features_upto_order
from gaugefixer.features.get_features_of_order import get_features_of_order
from gaugefixer.features.petti_feature import PettiFeature

def test_get_features_upto_order_basic():
    """Test basic functionality with simple inputs."""
    # Sequence length 2, max order 1, binary alphabet
    features = get_features_upto_order(L=2, max_order=1, alphabet=['A', 'B'])
    
    # Expected: order 0 feature + all order 1 features
    expected = [
        ((), ''),     # Order 0 feature
        ((0,), 'A'),  # Position 0, character A (order 1)
        ((0,), 'B'),  # Position 0, character B (order 1)
        ((1,), 'A'),  # Position 1, character A (order 1)
        ((1,), 'B'),  # Position 1, character B (order 1)
    ]
    
    assert features == expected
    assert len(features) == 5  # 1 (order 0) + 4 (order 1)

def test_get_features_upto_order_zero():
    """Test with max_order 0, which should return only the order 0 feature."""
    features = get_features_upto_order(L=3, max_order=0, alphabet=['A', 'B', 'C'])
    
    # Only order 0 feature expected
    expected = [((), '')]
    
    assert features == expected
    assert len(features) == 1

def test_get_features_upto_order_composition():
    """Test that the function correctly composes results from get_features_of_order."""
    L = 3
    alphabet = ['X', 'Y']
    
    # Get features for max_order = 2
    features_upto_2 = get_features_upto_order(L=L, max_order=2, alphabet=alphabet)
    
    # Get individual orders and combine them
    order_0 = get_features_of_order(L=L, order=0, alphabet=alphabet)
    order_1 = get_features_of_order(L=L, order=1, alphabet=alphabet)
    order_2 = get_features_of_order(L=L, order=2, alphabet=alphabet)
    
    # Combine and sort as in the function
    from gaugefixer.features.sort_features import sort_features
    combined = sort_features(order_0 + order_1 + order_2)
    
    assert features_upto_2 == combined
    
    # Check counts
    expected_count = (
        1 +  # Order 0: 1 feature (empty)
        (L * len(alphabet)) +  # Order 1: L positions * alphabet size
        (L * (L - 1) // 2 * len(alphabet) ** 2)  # Order 2: combinations of 2 positions * alphabet^2
    )
    
    assert len(features_upto_2) == expected_count

def test_get_features_upto_order_validation():
    """Test input validation."""
    # Test with invalid inputs
    with pytest.raises(AssertionError):
        get_features_upto_order(L=-1, max_order=1, alphabet=['A', 'B'])
    
    with pytest.raises(AssertionError):
        get_features_upto_order(L=5, max_order=-1, alphabet=['A', 'B'])
    
    with pytest.raises(AssertionError):
        get_features_upto_order(L=2, max_order=3, alphabet=['A', 'B'])
    
    with pytest.raises(AssertionError):
        get_features_upto_order(L=5, max_order=2, alphabet=[])
    
    # Test type checking (these should be caught by typeguard)
    with pytest.raises(Exception):  # Either AssertionError or TypeCheckError
        get_features_upto_order(L=5, max_order=2, alphabet=[1, 2, 3])

def test_get_features_upto_order_identical_to_single_order():
    """Test that max_order=n produces the same result as order=n when n=max_order."""
    L = 3
    order = 2
    alphabet = ['A', 'B']
    
    # Get features for single order
    features_single = get_features_of_order(L=L, order=order, alphabet=alphabet)
    
    # Get features up to order (should include all lower orders too)
    features_upto = get_features_upto_order(L=L, max_order=order, alphabet=alphabet)
    
    # The single order features should be a subset of the features_upto
    for feature in features_single:
        assert feature in features_upto
    
    # The features_upto should contain more features (from lower orders)
    assert len(features_upto) > len(features_single)

def test_get_features_upto_order_increasing_orders():
    """Test that increasing max_order increases the number of features."""
    L = 4
    alphabet = ['A', 'B']
    
    features_0 = get_features_upto_order(L=L, max_order=0, alphabet=alphabet)
    features_1 = get_features_upto_order(L=L, max_order=1, alphabet=alphabet)
    features_2 = get_features_upto_order(L=L, max_order=2, alphabet=alphabet)
    features_3 = get_features_upto_order(L=L, max_order=3, alphabet=alphabet)
    
    # Check increasing number of features
    assert len(features_0) < len(features_1) < len(features_2) < len(features_3)
    
    # Check that higher orders include all lower order features
    for f0 in features_0:
        assert f0 in features_1
        assert f0 in features_2
        assert f0 in features_3
    
    for f1 in features_1:
        assert f1 in features_2
        assert f1 in features_3
    
    for f2 in features_2:
        assert f2 in features_3

def test_get_features_upto_order_sorting():
    """Test that features are properly sorted."""
    features = get_features_upto_order(L=3, max_order=2, alphabet=['B', 'A'])
    
    # Check that positions are sorted first by length of positions tuple, 
    # then by position values, then by subsequence string
    for i in range(len(features) - 1):
        current = features[i]
        next_feature = features[i + 1]
        
        # Either the current position tuple is shorter
        if len(current[0]) < len(next_feature[0]):
            continue
        
        # Or they're the same length and current positions are "less than" next positions
        if len(current[0]) == len(next_feature[0]):
            if current[0] < next_feature[0]:
                continue
            
            # Or positions are equal and current subsequence is "less than" or equal to next
            if current[0] == next_feature[0] and current[1] <= next_feature[1]:
                continue
        
        # If we get here, sorting is incorrect
        assert False, f"Features not properly sorted: {current} before {next_feature}" 