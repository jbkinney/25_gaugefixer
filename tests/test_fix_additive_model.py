import pytest
import numpy as np
import pandas as pd

from gaugefixer.features import get_additive_features
from gaugefixer.fixers import fix_additive_model
from gaugefixer.verify import verify_identical_evaluation
from gaugefixer.sequence import SeqEmbedder, get_alphabet, evaluate_model_on_seqs, randseqs

class TestFixAdditiveModel:
    """Test class for testing fix_additive_model functionality."""
    
    def setup_method(self):
        """Setup test parameters."""
        self.L = 10  # Smaller L for faster tests
        self.alphabet = get_alphabet('dna')
        self.alpha = len(self.alphabet)
        self.features = get_additive_features(L=self.L, alphabet=self.alphabet)
        self.N = len(self.features)
        
        # Create random theta_series for testing
        self.theta_series = pd.Series(
            index=self.features,
            data=np.random.normal(size=self.N)
        )
    
    def test_fix_additive_model_zero_sum(self):
        """Test fix_additive_model with zero-sum gauge."""
        # Fix gauge with zero-sum
        fixed_theta_series = fix_additive_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum'
        )
        
        # Verify that the original and fixed parameters evaluate to the same values
        random_seqs = randseqs(num_seqs=10, L=self.L, alphabet=self.alphabet)
        embedder = SeqEmbedder(features=self.features, L=self.L)
        
        # Evaluate both sets of parameters
        orig_scores = evaluate_model_on_seqs(
            theta=self.theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        fixed_scores = evaluate_model_on_seqs(
            theta=fixed_theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        
        # Check if evaluations are equal
        np.testing.assert_allclose(orig_scores, fixed_scores, rtol=1e-10)
    
    def test_verify_identical_evaluation_direct(self):
        """Test using verify_identical_evaluation directly, similar to the notebook approach."""
        # Fix gauge with zero-sum
        fixed_theta_series = fix_additive_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum'
        )
        
        # Use the verify_identical_evaluation function directly
        # This will raise an exception if the evaluations don't match
        verify_identical_evaluation(
            theta1=self.theta_series,
            theta2=fixed_theta_series,
            L=self.L,
            alphabet=self.alphabet,
            num_tests=20,  # Test with more sequences
            verbose=False,
            raise_on_failure=True
        )
        
        # Also verify that features match
        assert all(self.theta_series.index == fixed_theta_series.index)
    
    def test_fix_additive_model_wild_type(self):
        """Test fix_additive_model with wild-type gauge."""
        # Create a wild-type sequence
        wt_seq = ''.join(np.random.choice(self.alphabet, size=self.L))
        
        # Fix gauge with wild-type
        fixed_theta_series = fix_additive_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='wild-type',
            wt_seq=wt_seq
        )
        
        # Verify that the original and fixed parameters evaluate to the same values
        random_seqs = randseqs(num_seqs=10, L=self.L, alphabet=self.alphabet)
        embedder = SeqEmbedder(features=self.features, L=self.L)
        
        # Evaluate both sets of parameters
        orig_scores = evaluate_model_on_seqs(
            theta=self.theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        fixed_scores = evaluate_model_on_seqs(
            theta=fixed_theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        
        # Check if evaluations are equal
        np.testing.assert_allclose(orig_scores, fixed_scores, rtol=1e-10)
    
    def test_fix_additive_model_custom_pi_lc(self):
        """Test fix_additive_model with custom pi_lc."""
        # Create a custom pi_lc (background frequencies)
        pi_lc = np.ones((self.L, self.alpha)) / self.alpha
        # Make it slightly non-uniform but still valid
        pi_lc = pi_lc + np.random.normal(0, 0.01, size=(self.L, self.alpha))
        # Ensure each row sums to 1
        pi_lc = pi_lc / pi_lc.sum(axis=1, keepdims=True)
        
        # Fix gauge with custom pi_lc
        fixed_theta_series = fix_additive_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge=None,
            pi_lc=pi_lc
        )
        
        # Verify that the original and fixed parameters evaluate to the same values
        random_seqs = randseqs(num_seqs=10, L=self.L, alphabet=self.alphabet)
        embedder = SeqEmbedder(features=self.features, L=self.L)
        
        # Evaluate both sets of parameters
        orig_scores = evaluate_model_on_seqs(
            theta=self.theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        fixed_scores = evaluate_model_on_seqs(
            theta=fixed_theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        
        # Check if evaluations are equal
        np.testing.assert_allclose(orig_scores, fixed_scores, rtol=1e-10)
    
    def test_fix_additive_model_numpy_input(self):
        """Test fix_additive_model with numpy array input and features."""
        # Convert Series to numpy array and keep features separate
        theta_array = self.theta_series.values
        
        # Fix gauge with numpy array input and features
        fixed_theta_series = fix_additive_model(
            theta=theta_array,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum',
            features=self.features
        )
        
        # Verify that the original and fixed parameters evaluate to the same values
        random_seqs = randseqs(num_seqs=10, L=self.L, alphabet=self.alphabet)
        embedder = SeqEmbedder(features=self.features, L=self.L)
        
        # Evaluate both sets of parameters - convert fixed_theta_series to a series with the correct features
        orig_scores = evaluate_model_on_seqs(
            theta=self.theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        fixed_scores = evaluate_model_on_seqs(
            theta=fixed_theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        
        # Check if evaluations are equal
        np.testing.assert_allclose(orig_scores, fixed_scores, rtol=1e-10)
    
    def test_fix_additive_model_numpy_input_no_features(self):
        """Test fix_additive_model with numpy array input and no features."""
        # Convert Series to numpy array
        theta_array = self.theta_series.values
        sorted_features = get_additive_features(L=self.L, alphabet=self.alphabet)
        
        # Fix gauge with numpy array input and no features
        fixed_theta_series = fix_additive_model(
            theta=theta_array,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum'
        )
        
        # Verify that the original and fixed parameters evaluate to the same values
        random_seqs = randseqs(num_seqs=10, L=self.L, alphabet=self.alphabet)
        embedder = SeqEmbedder(features=sorted_features, L=self.L)
        
        # Evaluate both sets of parameters
        orig_scores = evaluate_model_on_seqs(
            theta=pd.Series(index=sorted_features, data=theta_array),
            seqs=random_seqs, 
            embedder=embedder
        )
        fixed_scores = evaluate_model_on_seqs(
            theta=fixed_theta_series,
            seqs=random_seqs, 
            embedder=embedder
        )
        
        # Check if evaluations are equal
        np.testing.assert_allclose(orig_scores, fixed_scores, rtol=1e-10)
    
    def test_invalid_input_combinations(self):
        """Test invalid input combinations."""
        # Invalid gauge type
        with pytest.raises(AssertionError):
            fix_additive_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='invalid-gauge'
            )
        
        # Missing wild-type sequence
        with pytest.raises(AssertionError):
            fix_additive_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='wild-type'
            )
        
        # Invalid wild-type sequence length
        with pytest.raises(AssertionError):
            fix_additive_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='wild-type',
                wt_seq='A'  # Too short
            )
        
        # Invalid wild-type sequence characters
        with pytest.raises(AssertionError):
            invalid_seq = 'A' * (self.L - 1) + 'X'  # 'X' not in alphabet
            fix_additive_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='wild-type',
                wt_seq=invalid_seq
            )
        
        # Missing pi_lc for custom gauge
        with pytest.raises(AssertionError):
            fix_additive_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge=None
            )
        
        # Invalid pi_lc shape
        with pytest.raises(AssertionError):
            invalid_pi_lc = np.ones((self.L - 1, self.alpha)) / self.alpha  # Wrong L
            fix_additive_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge=None,
                pi_lc=invalid_pi_lc
            )
    
    def test_theta_and_features_mismatch(self):
        """Test when theta and features don't match."""
        # Mismatched theta length and features length
        with pytest.raises(AssertionError):
            wrong_theta = np.random.normal(size=self.N - 1)  # Wrong length
            fix_additive_model(
                theta=wrong_theta,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='zero-sum',
                features=self.features
            )
        
        # Mismatched features type
        subset_features = self.features[:-1]  # Different set of features
        with pytest.raises(AssertionError):
            fix_additive_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='zero-sum',
                features=subset_features
            ) 