import pandas as pd
import numpy as np
from gaugefixer.sequence.seq_embedder import SeqEmbedder

def evaluate_model_on_seqs(theta: pd.Series,
                            seqs: list[str],
                            embedder: SeqEmbedder | None = None,
                            L: int | None = None,
                            alphabet: list[str] | None = None,
                            ) -> pd.Series:
    
    match (embedder, L, alphabet):
        case (SeqEmbedder(), None, None):
            pass;
        case (None, int(), list()):
            print("Creating embedder")
            embedder = SeqEmbedder(features=list(theta.index), L=L, alphabet=alphabet)
        case _:
            raise ValueError("Invalid input")
    
    
    X = np.array([embedder.embed(seq) for seq in seqs])
    f = X@theta.values
    return f 