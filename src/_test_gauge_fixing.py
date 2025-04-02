from random import choices
import numpy as np
from src.seq_embedder import SeqEmbedder

def _test_gauge_fixing(theta_series, theta_fixed_series, L, alphabet, embedder=None, num_trials=100):

    # Check that features match
    assert len(theta_series) == len(theta_fixed_series)
    assert theta_series.index.equals(theta_fixed_series.index)
    features = list(theta_series.index)

    # Create embedder if not provided
    if embedder is None:
        embedder = SeqEmbedder(features=features, L=L)

    # Test that the two vectors produce the same function values
    for trial_num in range(num_trials):
        seq = ''.join(choices(alphabet, k=L))

        # Get embedded sequence
        x =embedder.embed(seq)

        # Compute function using the two vectors
        f = theta_series@x
        f_fixed = theta_fixed_series@x
        
        # Check that the two functions are close
        assert np.isclose(f, f_fixed), f'{f=}\n{f_fixed=}'
        print(f'{trial_num:03d} {f=:f} \t {f_fixed=:f}')
        
    print(f'Tested {num_trials} random sequences; all seqs passed.')