from src.get_augseqs_upto_order import get_augseqs_upto_order
from src.get_adjacent_orbits_of_order import get_adjacent_orbits_of_order
from functools import partial

get_additive_augseqs = partial(get_augseqs_upto_order, max_order=1)
get_pairwise_augseqs = partial(get_augseqs_upto_order, max_order=2)
get_neighbor_augseqs = partial(get_adjacent_orbits_of_order, max_order=2)


