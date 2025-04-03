import pytest
import numpy as np
import pandas as pd

from gaugefixer.features import get_allorder_features
from gaugefixer.fixers import fix_allorder_model
from gaugefixer.verify import verify_identical_evaluation, verify_marginalization
from gaugefixer.sequence import SeqEmbedder, get_alphabet, evaluate_model_on_seqs, randseqs, randseq

class TestFixAllorderModel:
    """Test class for testing fix_allorder_model functionality."""
    
    def setup_method(self):
        """Setup test parameters."""
        # Use a smaller L for faster tests
        self.L = 5  # Much smaller than the notebook's L=9 for speed
        self.alphabet = get_alphabet('dna')
        self.alpha = len(self.alphabet)
        self.features = get_allorder_features(L=self.L, alphabet=self.alphabet)
        self.N = len(self.features)
        
        # Create random theta_series for testing
        self.theta_series = pd.Series(
            index=self.features,
            data=np.random.normal(size=self.N)
        )
    
    def test_fix_allorder_model_zero_sum(self):
        """Test fix_allorder_model with zero-sum gauge."""
        # Fix gauge with zero-sum
        fixed_theta_series = fix_allorder_model(
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
        """Test using verify_identical_evaluation directly."""
        # Fix gauge with zero-sum
        fixed_theta_series = fix_allorder_model(
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
            num_tests=10,  # Test with fewer sequences for speed
            verbose=False,
            raise_on_failure=True
        )
        
        # Also verify that features match
        assert set(self.theta_series.index) == set(fixed_theta_series.index)
    
    def test_fix_allorder_model_wild_type(self):
        """Test fix_allorder_model with wild-type gauge."""
        # Create a wild-type sequence
        wt_seq = ''.join(np.random.choice(self.alphabet, size=self.L))
        
        # Fix gauge with wild-type
        fixed_theta_series = fix_allorder_model(
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
    
    def test_verify_marginalization_wild_type(self):
        """Test that wild-type gauge fixing enforces marginalization conditions."""
        # Create a wild-type sequence
        wt_seq = randseq(L=self.L, alphabet=self.alphabet)
        
        # Fix gauge with wild-type
        fixed_theta_series = fix_allorder_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='wild-type',
            wt_seq=wt_seq
        )
        
        # Use the verify_marginalization function to check the marginalization property
        # This will verify that the model exhibits the expected marginalization behavior
        verify_marginalization(
            theta_series=fixed_theta_series,
            num_tests=20,  # Test with fewer tests for speed
            L=self.L,
            alphabet=self.alphabet,
            gauge='wild-type',
            wt_seq=wt_seq
        )
    
    def test_fix_allorder_model_hierarchical(self):
        """Test fix_allorder_model with hierarchical gauge."""
        # Create custom pi_lc for hierarchical gauge
        pi_lc = np.ones((self.L, self.alpha)) / self.alpha
        
        # Fix gauge with hierarchical
        fixed_theta_series = fix_allorder_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='hierarchical',
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
    
    def test_fix_allorder_model_trivial(self):
        """Test fix_allorder_model with trivial gauge (lambda=0)."""
        # Fix gauge with trivial
        fixed_theta_series = fix_allorder_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='trivial'
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
    
    def test_fix_allorder_model_euclidean(self):
        """Test fix_allorder_model with euclidean gauge (lambda=1)."""
        # Fix gauge with euclidean
        fixed_theta_series = fix_allorder_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='euclidean'
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
    
    def test_fix_allorder_model_equitable(self):
        """Test fix_allorder_model with equitable gauge (lambda=|alphabet|)."""
        # Fix gauge with equitable
        fixed_theta_series = fix_allorder_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='equitable'
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
    
    def test_fix_allorder_model_custom_lambda_pi_lc(self):
        """Test fix_allorder_model with custom lambda and pi_lc."""
        # Create a custom pi_lc and lambda
        pi_lc = np.ones((self.L, self.alpha)) / self.alpha
        # Make it slightly non-uniform but still valid
        pi_lc = pi_lc + np.random.normal(0, 0.01, size=(self.L, self.alpha))
        # Ensure each row sums to 1
        pi_lc = pi_lc / pi_lc.sum(axis=1, keepdims=True)
        lam = 0.5  # Custom lambda value
        
        # Fix gauge with custom lambda and pi_lc
        fixed_theta_series = fix_allorder_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge=None,
            lam=lam,
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
    
    def test_fix_allorder_model_numpy_input(self):
        """Test fix_allorder_model with numpy array input and features."""
        # Convert Series to numpy array and keep features separate
        theta_array = self.theta_series.values
        
        # Fix gauge with numpy array input and features
        fixed_theta_series = fix_allorder_model(
            theta=theta_array,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum',
            features=self.features
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
    
    def test_fix_allorder_model_numpy_input_no_features(self):
        """Test fix_allorder_model with numpy array input and no features."""
        # Convert Series to numpy array
        theta_array = self.theta_series.values
        
        # Fix gauge with numpy array input and no features
        fixed_theta_series = fix_allorder_model(
            theta=theta_array,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum'
        )
        
        # Since we can't use verify_identical_evaluation directly due to different feature orderings,
        # we'll manually verify that the evaluations are the same
        random_seqs = randseqs(num_seqs=20, L=self.L, alphabet=self.alphabet)
        
        # Create embedders for each feature ordering
        embedder_original = SeqEmbedder(features=list(self.theta_series.index), L=self.L)
        embedder_fixed = SeqEmbedder(features=list(fixed_theta_series.index), L=self.L)
        
        # Compare evaluations on random sequences
        for seq in random_seqs:
            # Get embedded sequences
            x_orig = embedder_original.embed(seq)
            x_fixed = embedder_fixed.embed(seq)
            
            # Compute function values
            f_orig = self.theta_series @ x_orig
            f_fixed = fixed_theta_series @ x_fixed
            
            # Check that the function values are equal
            np.testing.assert_allclose(f_orig, f_fixed, rtol=1e-10)
        
        # Also verify feature set equality (but not order)
        assert set(self.theta_series.index) == set(fixed_theta_series.index)
    
    def test_fix_allorder_model_feature_order_preservation(self):
        """Test that the feature order is preserved after fixing."""
        # Fix gauge with zero-sum
        fixed_theta_series = fix_allorder_model(
            theta=self.theta_series,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum'
        )
        
        # Verify the feature order is preserved
        assert list(self.theta_series.index) == list(fixed_theta_series.index)
        
        # Reorder the features to test order preservation
        shuffled_idx = np.random.permutation(len(self.features))
        shuffled_features = [self.features[i] for i in shuffled_idx]
        shuffled_theta = pd.Series(
            index=shuffled_features,
            data=self.theta_series.values[shuffled_idx]
        )
        
        # Fix gauge with reordered features
        fixed_shuffled_theta = fix_allorder_model(
            theta=shuffled_theta,
            L=self.L, 
            alphabet=self.alphabet,
            gauge='zero-sum'
        )
        
        # Verify that the order is preserved after fixing
        assert list(shuffled_theta.index) == list(fixed_shuffled_theta.index)
    
    def test_invalid_input_combinations(self):
        """Test invalid input combinations."""
        # Invalid gauge type
        with pytest.raises(AssertionError):
            fix_allorder_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='invalid-gauge'
            )
        
        # Missing wild-type sequence
        with pytest.raises(AssertionError):
            fix_allorder_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='wild-type'
            )
        
        # Invalid wild-type sequence length
        with pytest.raises(AssertionError):
            fix_allorder_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='wild-type',
                wt_seq='A'  # Too short
            )
        
        # Invalid wild-type sequence characters
        with pytest.raises(AssertionError):
            invalid_seq = 'A' * (self.L - 1) + 'X'  # 'X' not in alphabet
            fix_allorder_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='wild-type',
                wt_seq=invalid_seq
            )
        
        # Missing pi_lc for hierarchical gauge
        with pytest.raises(AssertionError):
            fix_allorder_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='hierarchical'
            )
        
        # Missing lambda for custom gauge
        with pytest.raises(AssertionError):
            fix_allorder_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge=None,
                pi_lc=np.ones((self.L, self.alpha)) / self.alpha
            )
        
        # Invalid lambda value
        with pytest.raises(AssertionError):
            fix_allorder_model(
                theta=self.theta_series,
                L=self.L, 
                alphabet=self.alphabet,
                gauge=None,
                lam=-1.0,  # Negative lambda
                pi_lc=np.ones((self.L, self.alpha)) / self.alpha
            )
    
    def test_theta_and_features_mismatch(self):
        """Test when theta and features don't match."""
        # Mismatched theta length and features length
        with pytest.raises(AssertionError):
            wrong_theta = np.random.normal(size=self.N - 1)  # Wrong length
            fix_allorder_model(
                theta=wrong_theta,
                L=self.L, 
                alphabet=self.alphabet,
                gauge='zero-sum',
                features=self.features
            )
        
        # Features set doesn't match the expected all-order features
        # Create a different set of features
        from itertools import product
        augalphabet = ['*'] + self.alphabet
        fake_augseqs = [''.join(seq) for seq in product(augalphabet, repeat=3)]  # Using wrong L
        fake_features = []
        for augseq in fake_augseqs:
            orbit = tuple(i for i, c in enumerate(augseq) if c != '*')
            subseq = ''.join(c for c in augseq if c != '*')
            fake_features.append((orbit, subseq))
            
        # Test with mismatched features
        wrong_theta = np.random.normal(size=len(fake_features))
        with pytest.raises(AssertionError):
            fix_allorder_model(
                theta=wrong_theta,
                L=self.L,  # L=5, but features are for L=3
                alphabet=self.alphabet,
                gauge='zero-sum',
                features=fake_features
            ) 