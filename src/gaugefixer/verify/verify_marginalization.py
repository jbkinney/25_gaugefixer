import numpy as np
import pandas as pd
import random

def verify_marginalization(
    theta_series: pd.Series,
    L: int,
    alphabet: list[str],
    gauge: str | None = None,
    pi_lc: np.ndarray | None = None,
    lam: float | None = None,
    wt_seq: str | None = None,
    num_tests: int = 100,
    verbose: bool = True,
    raise_on_failure: bool = True):
    
    # Copy input theta and convert to multi-index series
    theta = theta_series.copy()
    features = theta.index
    theta.index = pd.MultiIndex.from_tuples(features)
    
    alpha = len(alphabet)
    
    # Handle different gauge fixing cases
    match (gauge, lam, pi_lc, wt_seq):
        
        case ('wild-type', None, None, str()):
            assert len(wt_seq) == L
            assert set(wt_seq) <= set(alphabet)
            lam = np.inf
            pi_lc = np.array([[c==wt_c for j,c in enumerate(alphabet)] for i,wt_c in enumerate(wt_seq)])
            
        case ('zero-sum', None, None, None):
            lam = np.inf
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case ('hierarchical', None, np.ndarray(), None):
            assert pi_lc.shape == (L, alpha)
            lam = np.inf
            
        case ('trivial', None, None, None):
            lam = 0
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case ('euclidean', None, None, None):
            lam = 1
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case ('equitable', None, None, None):
            lam = alpha
            pi_lc = np.ones(shape=(L,alpha))/alpha
            
        case (None, float(), np.ndarray(), None):
            pass
        
        case _:
            assert False, f'Invalid combination of inputs {gauge=}, {lam=}, {pi_lc=}, {wt_seq=}.'
    
    # Get list of up to max_num_orders orders
    all_orders = list(set([len(orb) for orb, _ in features if len(orb) > 0]))
    
    # lam = 0 has to be treated separately
    if lam == 0:
        zero_ix = np.where([len(orb)<L for orb, _ in features])[0]
        result = np.all([theta.iloc[zero_ix] == 0])
        if verbose: 
            if result:
                print('Passed: all terms of order < L are zero')
            else:
                print('FAILED! Not all terms of order < L are zero')
        if raise_on_failure and not result:
            raise ValueError('Failed: all terms of order < L are not zero')
        return None
    
    # For each order, perform up to max_tests_per_order tests
    for test_num in range(num_tests):
        # Choose random order
        order = random.choice(all_orders)
        
        # Get random orbit of order ord
        orbit = random.choice(list(set([orb for orb, _ in features if len(orb)==order])))
        
        # Choose random wt sequence 
        wtseq = np.random.choice([s for o, s in features if o==orbit])
        
        # Choose random position in sequence
        pos = random.randrange(order)
        
        # Get list of sequences to sum
        seqs_to_sum = [wtseq[:pos] + c + wtseq[pos+1:] for c in alphabet]
        seq_weights = np.array([pi_lc[orbit[pos], j] for j in range(alpha)])
        seq_values = np.array([theta.loc[(orbit, seq)] for seq in seqs_to_sum])
        
        # Sum over sequences to sum
        lhs_result = sum(seq_values * seq_weights)
        
        rhs_orbit = orbit[:pos] + orbit[pos+1:]
        rhs_seq = wtseq[:pos] + wtseq[pos+1:]
        rhs_result = theta.loc[(rhs_orbit, rhs_seq)]/lam if np.isfinite(lam) else 0
        
        # Get sequence to show
        seq_to_show = wtseq[:pos] + '*' + wtseq[pos+1:]
        
        # Print result
        if verbose:
            message = 'Passed' if np.isclose(lhs_result, rhs_result) else f'FAILED! {lhs_result=}, {rhs_result=}'
            print(f'{test_num=}: {order=}, {orbit=}, {pos=}, {seq_to_show=}, {message}')
            
        if raise_on_failure and not np.isclose(lhs_result, rhs_result):
            raise ValueError('Failed: marginalization property does not hold')
        
    return None