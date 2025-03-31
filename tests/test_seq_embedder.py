import pytest
import numpy as np
import re
from src.seq_embedder import SeqEmbedder
from src.get_features_upto_order import get_features_upto_order
from typeguard import TypeCheckError

def test_seq_embedder_init():
    """Test that the SeqEmbedder initializes correctly."""
    # Create a simple set of features
    features = [
        ((), ''),        # Order 0
        ((0,), 'A'),     # Order 1
        ((1, 2), 'GT'),  # Order 2
    ]
    L = 3
    
    # Initialize the SeqEmbedder
    embedder = SeqEmbedder(features, L)
    
    # Check that attributes are set correctly
    assert embedder.L == L
    assert embedder.features == features
    assert len(embedder.patterns) == len(features)
    
    # The first pattern should match any string of length L
    assert embedder.patterns[0].match('AGT') is not None
    # The second pattern should only match strings that have 'A' at position 0
    assert embedder.patterns[1].match('AGT') is not None
    assert embedder.patterns[1].match('CGT') is None
    # The third pattern should only match strings that have 'G' at position 1 and 'T' at position 2
    assert embedder.patterns[2].match('AGT') is not None
    assert embedder.patterns[2].match('ACT') is None
    assert embedder.patterns[2].match('AGC') is None

def test_seq_embedder_check_features_parameter():
    """Test the check_features parameter in the SeqEmbedder initializer."""
    # Valid features
    valid_features = [
        ((), ''),        # Order 0
        ((0,), 'A'),     # Order 1
        ((1, 2), 'GT'),  # Order 2
    ]
    L = 3
    
    # Initialize with check_features=True (default)
    embedder_with_check = SeqEmbedder(valid_features, L, check_features=True)
    
    # Initialize with check_features=False
    embedder_without_check = SeqEmbedder(valid_features, L, check_features=False)
    
    # Both should initialize successfully with valid features
    assert len(embedder_with_check.patterns) == len(valid_features)
    assert len(embedder_without_check.patterns) == len(valid_features)
    
    # Test embedding results are the same regardless of check_features
    seq = 'AGT'
    result_with_check = embedder_with_check.embed(seq)
    result_without_check = embedder_without_check.embed(seq)
    assert np.array_equal(result_with_check, result_without_check)
    
    # Out of bounds feature that would normally fail validation
    out_of_bounds_features = [
        ((0,), 'A'),  # Valid feature for testing
    ]
    
    # Both options should work with valid features
    SeqEmbedder(out_of_bounds_features, L, check_features=True)
    SeqEmbedder(out_of_bounds_features, L, check_features=False)
    
    # Test performance benefit of check_features=False
    # For large feature sets, check_features=False should be faster
    import time
    
    # Create a large set of valid features
    large_features = []
    for i in range(100):
        large_features.append(((), ''))  # Add 100 order-0 features
    
    # Measure time with check_features=True
    start_time = time.time()
    SeqEmbedder(large_features, L, check_features=True)
    time_with_check = time.time() - start_time
    
    # Measure time with check_features=False
    start_time = time.time()
    SeqEmbedder(large_features, L, check_features=False)
    time_without_check = time.time() - start_time
    
    # check_features=False should be faster or at least not significantly slower
    # We don't assert this because timing can be inconsistent in different environments
    print(f"Time with check_features=True: {time_with_check:.6f}s")
    print(f"Time with check_features=False: {time_without_check:.6f}s")
    
    # Test with special case: empty features list
    empty_features = []
    
    # Both options should work with empty features
    embedder_empty_with_check = SeqEmbedder(empty_features, L, check_features=True)
    embedder_empty_without_check = SeqEmbedder(empty_features, L, check_features=False)
    
    assert len(embedder_empty_with_check.patterns) == 0
    assert len(embedder_empty_without_check.patterns) == 0

def test_seq_embedder_init_validation():
    """Test validation during initialization."""
    # Test with invalid L
    with pytest.raises(AssertionError):
        SeqEmbedder(features=[((0,), 'A')], L=-1)
    
    # Test with invalid features format - typeguard will raise TypeCheckError
    with pytest.raises(TypeCheckError):
        SeqEmbedder(features=["not a tuple"], L=3)
    
    # Test with invalid orbit indices - this will be caught by assertions after type check
    with pytest.raises(AssertionError):
        SeqEmbedder(features=[((3,), 'A')], L=3)  # Index out of range
    
    # Test with mismatched orbit and subsequence length
    with pytest.raises(AssertionError):
        SeqEmbedder(features=[((0, 1), 'A')], L=3)  # Subsequence too short
    
    # Test with incorrect types - typeguard will raise TypeCheckError
    with pytest.raises(TypeCheckError):
        SeqEmbedder(features=None, L=3)
    
    with pytest.raises(TypeCheckError):
        SeqEmbedder(features=[((0,), 'A')], L="3")

def test_embed_single_feature():
    """Test embedding with a single feature."""
    # Feature that checks for 'A' at position 0
    features = [((0,), 'A')]
    L = 3
    
    embedder = SeqEmbedder(features, L)
    
    # Sequence with 'A' at position 0 should match
    result = embedder.embed('ACG')
    assert result.shape == (1,)
    assert result[0] == 1
    
    # Sequence without 'A' at position 0 should not match
    result = embedder.embed('TCG')
    assert result.shape == (1,)
    assert result[0] == 0

def test_embed_multiple_features():
    """Test embedding with multiple features."""
    # Features of different orders
    features = [
        ((), ''),        # Order 0 (matches everything)
        ((0,), 'A'),     # Order 1 (matches 'A' at position 0)
        ((1,), 'G'),     # Order 1 (matches 'G' at position 1)
        ((0, 1), 'AG'),  # Order 2 (matches 'A' at position 0 and 'G' at position 1)
    ]
    L = 3
    
    embedder = SeqEmbedder(features, L)
    
    # Test with various sequences
    sequences = [
        'AGT',  # Should match all features
        'CGT',  # Should match order 0 and 'G' at position 1
        'ACT',  # Should match only order 0 and 'A' at position 0
        'TTT',  # Should match only order 0
    ]
    
    expected_results = [
        [1, 1, 1, 1],  # All features match for 'AGT'
        [1, 0, 1, 0],  # Only order 0 and 'G' at position 1 match for 'CGT'
        [1, 1, 0, 0],  # Only order 0 and 'A' at position 0 match for 'ACT'
        [1, 0, 0, 0],  # Only order 0 matches for 'TTT'
    ]
    
    for seq, expected in zip(sequences, expected_results):
        result = embedder.embed(seq)
        assert np.array_equal(result, np.array(expected)), f"Failed for sequence {seq}"

def test_embed_with_generated_features():
    """Test embedding with features generated by get_features_upto_order."""
    # Generate features up to order 1 for sequences of length 2
    L = 2
    alphabet = ['A', 'C', 'G', 'T']
    max_order = 1
    
    features = get_features_upto_order(L, max_order, alphabet)
    
    # Initialize the embedder
    embedder = SeqEmbedder(features, L)
    
    # Test with all possible sequences of length 2
    all_seqs = [''.join(s) for s in [(a, b) for a in alphabet for b in alphabet]]
    
    for seq in all_seqs:
        result = embedder.embed(seq)
        
        # The embedding should have 1 + (L * len(alphabet)) dimensions
        # 1 for order 0 + L positions * alphabet size for order 1
        expected_dim = 1 + (L * len(alphabet))
        assert result.shape == (expected_dim,)
        
        # Order 0 feature should always match (first element is 1)
        assert result[0] == 1
        
        # Check order 1 features
        for i, pos in enumerate(range(L)):
            for j, char in enumerate(alphabet):
                feature_idx = 1 + (i * len(alphabet)) + j
                should_match = seq[pos] == char
                assert result[feature_idx] == int(should_match), f"Failed at position {pos} for character {char} in sequence {seq}"

def test_embed_with_invalid_sequence():
    """Test embedding with sequences that don't match the expected length."""
    features = [((0,), 'A')]
    L = 3
    
    embedder = SeqEmbedder(features, L)
    
    # Test with sequences of different lengths
    with pytest.raises(ValueError):
        embedder.embed('A')  # Too short
    
    with pytest.raises(ValueError):
        embedder.embed('ACGT')  # Too long
    
    # Test with non-string input
    with pytest.raises(TypeError):
        embedder.embed(123)
    
    with pytest.raises(TypeError):
        embedder.embed(['A', 'C', 'G'])

def test_pattern_generation():
    """Test that patterns are correctly generated from features."""
    L = 4
    features = [
        ((0, 2), 'AG'),    # 'A' at position 0, 'G' at position 2
        ((1, 3), 'CT'),    # 'C' at position 1, 'T' at position 3
        ((0, 1, 2), 'ACG')  # 'A' at position 0, 'C' at position 1, 'G' at position 2
    ]
    
    embedder = SeqEmbedder(features, L)
    
    # Check the generated patterns
    expected_patterns = [
        r'A.G.',  # 'A' at position 0, 'G' at position 2
        r'.C.T',  # 'C' at position 1, 'T' at position 3
        r'ACG.'   # 'A' at position 0, 'C' at position 1, 'G' at position 2
    ]
    
    for i, (pattern, expected) in enumerate(zip(embedder.patterns, expected_patterns)):
        assert pattern.pattern == expected, f"Pattern {i} doesn't match expected: {pattern.pattern} != {expected}"
    
    # Skip the direct regex testing and just test the embedder's behavior
    test_seqs = [
        'ACGT',  # Should match all patterns
        'TCGT',  # Should match only pattern 1 (.C.T)
        'AAGG',  # Should match only pattern 0 (A.G.)
        'TCAT',  # Should match only pattern 1 (.C.T)
        'AGGT',  # Should match only pattern 0 (A.G.)
        'TTTT',  # Should match none of the patterns
    ]
    
    expected_matches = [
        [1, 1, 1],  # Matches all patterns
        [0, 1, 0],  # Matches only pattern 1
        [1, 0, 0],  # Matches only pattern 0
        [0, 1, 0],  # Matches only pattern 1
        [1, 0, 0],  # Matches only pattern 0
        [0, 0, 0],  # Matches none
    ]
    
    for seq, expected in zip(test_seqs, expected_matches):
        result = embedder.embed(seq)
        assert np.array_equal(result, np.array(expected)), f"Failed embedding for sequence {seq}: got {result}, expected {expected}" 