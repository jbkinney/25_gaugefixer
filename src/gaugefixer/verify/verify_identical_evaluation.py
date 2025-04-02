from random import choices
import numpy as np
from gaugefixer.seq_embedder import SeqEmbedder
import pandas as pd
def verify_identical_evaluation(
    theta1: pd.Series,
    theta2: pd.Series,
    L: int,
    alphabet: list[str],
    embedder: SeqEmbedder | None = None,
    num_tests: int = 100,
    verbose: bool = True,
    raise_on_failure: bool = True):

    # Check that features match
    assert len(theta1) == len(theta2)
    assert theta1.index.equals(theta2.index)
    features = list(theta1.index)

    # Create embedder if not provided
    if embedder is None:
        embedder = SeqEmbedder(features=features, L=L)

    # Test that the two vectors produce the same function values
    for test_num in range(num_tests):
        seq = ''.join(choices(alphabet, k=L))

        # Get embedded sequence
        x =embedder.embed(seq)

        # Compute function using the two vectors
        f = theta1@x
        f_fixed = theta2@x
        
        # Check that the two functions are close
        if verbose:
            message = 'Passed' if np.isclose(f, f_fixed) else f'FAILED! {f=}\n{f_fixed=}'
            print(f'{test_num:03d} {f=:f} \t {f_fixed=:f}, {message}')
        if raise_on_failure and not np.isclose(f, f_fixed):
            raise ValueError('Failed: the two functions are not close')
        
    print(f'Tested {num_tests} random sequences; all seqs passed.') 