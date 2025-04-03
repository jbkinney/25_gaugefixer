import pytest
from gaugefixer.features.sort_features import sort_features
from gaugefixer.features.petti_feature import PettiFeature

class TestSortFeatures:
    def test_empty_list(self):
        """Test sorting an empty list of features."""
        assert sort_features([]) == []
        
    def test_single_feature(self):
        """Test sorting a list with a single feature."""
        features = [((1, 2, 3), "abc")]
        assert sort_features(features) == features
        
    def test_sort_by_orbit_size(self):
        """Test sorting features by the size of the orbit tuples."""
        features = [
            ((1, 2, 3), "abc"),  # size 3
            ((1,), "x"),         # size 1
            ((1, 2), "ab")       # size 2
        ]
        expected = [
            ((1,), "x"),         # size 1
            ((1, 2), "ab"),      # size 2
            ((1, 2, 3), "abc")   # size 3
        ]
        assert sort_features(features) == expected
        
    def test_sort_by_orbit_values(self):
        """Test sorting features with same-sized orbits by their values."""
        features = [
            ((2, 3), "xy"),
            ((1, 3), "pq"),
            ((1, 2), "ab")
        ]
        expected = [
            ((1, 2), "ab"),
            ((1, 3), "pq"),
            ((2, 3), "xy")
        ]
        assert sort_features(features) == expected
        
    def test_sort_by_label(self):
        """Test sorting features with identical orbits by their labels."""
        features = [
            ((1, 2), "xyz"),
            ((1, 2), "abc"),
            ((1, 2), "def")
        ]
        expected = [
            ((1, 2), "abc"),
            ((1, 2), "def"),
            ((1, 2), "xyz")
        ]
        assert sort_features(features) == expected
        
    def test_complex_sorting(self):
        """Test sorting with all criteria in play."""
        features = [
            ((2, 3), "a"),       # size 2, values (2,3)
            ((1, 2, 3), "xyz"),  # size 3
            ((1,), "b"),         # size 1
            ((1, 2), "def"),     # size 2, values (1,2)
            ((1, 2), "abc"),     # size 2, values (1,2)
            ((1, 3), "pq")       # size 2, values (1,3)
        ]
        expected = [
            ((1,), "b"),         # size 1
            ((1, 2), "abc"),     # size 2, values (1,2), label "abc"
            ((1, 2), "def"),     # size 2, values (1,2), label "def"
            ((1, 3), "pq"),      # size 2, values (1,3)
            ((2, 3), "a"),       # size 2, values (2,3)
            ((1, 2, 3), "xyz")   # size 3
        ]
        assert sort_features(features) == expected 