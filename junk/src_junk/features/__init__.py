from src.features.get_features_upto_order import get_features_upto_order

def get_additive_features(
    L: int,
    alphabet: list[str]):
    return get_features_upto_order(L=L, 
                                   max_order=1, 
                                   alphabet=alphabet)
    
def get_pairwise_features(
    L: int,
    alphabet: list[str]):
    return get_features_upto_order(L=L, 
                                   max_order=2, 
                                   alphabet=alphabet)
    
def get_allorder_features(
    L: int,
    alphabet: list[str]):
    return get_features_upto_order(L=L, 
                                   max_order=L, 
                                   alphabet=alphabet)

def get_Korder_features(
    L: int,
    K: int,
    alphabet: list[str]):
    return get_features_upto_order(L=L, 
                                   max_order=K, 
                                   alphabet=alphabet)
