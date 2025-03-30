# benchmark.py
import cProfile
import pstats
from src.get_hg_projection_matrix import get_hg_projection_matrix
from src.seq_embedder import SeqEmbedder
from src import get_alphabet, get_pairwise_model_features

# Import other required functions

if __name__ == '__main__':
    # Make Olson et al. features
    L = 10 #55
    alphabet = get_alphabet('protein')
    num_to_show = 10
    wt_seq = 'QYKLI LNGKT LKGET TTGAV DAATA EKVFK QYAND NGVDG EWTYD DATKT FTVTE'.replace(' ', '')[:L]
    assert len(wt_seq) == L, f'{len(wt_seq)=} != {L=}'
    features = get_pairwise_model_features(L=L, alphabet=alphabet)
    N = len(features)
    print(f'{L=}\n{N=:,}\n{alphabet=}')
    print(f'all:\n{features[:num_to_show]=}\n{features[-num_to_show:]=}\n')

    # Default keyword arguments that are common across all background types
    default_kwargs = {
        'features': features,
        'alphabet': alphabet, 
        'wildcard_char': '*',
        'out_type': 'sparse'
    }

    # Variable keyword arguments for different background types
    bg_type_kwargs = {
        'wildtype': {'bg_type': 'wildtype', 'wt_seq': wt_seq},
        'uniform': {'bg_type': 'uniform'},
        #'custom': {'bg_type': 'custom', 'bg_df': bg_df},
    }

    # Update each params dict with the default kwargs
    for bg_type in bg_type_kwargs:
        bg_type_kwargs[bg_type].update(default_kwargs)

    # Loop over background types
    for bg_type, kwargs in bg_type_kwargs.items():
        print(f"\nProfiling with {bg_type} background:")
        with cProfile.Profile() as profiler:
            P_df = get_hg_projection_matrix(**kwargs)
            
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats('cumulative').print_stats(10)