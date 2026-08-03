import numpy as np
import pandas as pd
from gaugefixer import AllOrderModel
from gaugefixer.utils import get_orbits_features, get_subsets_of_set

if __name__ == "__main__":
    print("Loading Shine-Dalgarno landscape")
    function = pd.read_csv("data/shine_dalgarno.csv", index_col=0)

    print("Initializing AllOrderModel")
    model = AllOrderModel(L=9, alphabet_name="rna")
    model.set_landscape(function["f"])

    pi_uniform = [0.25 * np.ones(4)]
    pi_motif_1 = [
        np.array([0, 0, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
    ]
    pi_motif_2 = [
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
    ]
    motif_labels = ['GAGGAG', 'AGGAGU']
    motif_positions = [[-12, -11, -10, -9], [-13, -12, -11, -10]]

    theta0 = []
    for motif_label, pi_motif, positions in zip(motif_labels, [pi_motif_1, pi_motif_2], motif_positions):
        print(f"Analyzing {motif_label} motif at each position")
        print("  Fixing the gauge around the motif at each position")
        thetas = {}
        for p, position in enumerate(positions):
            print(f"\tPosition {p}")
            pi_lc = pi_uniform * p + pi_motif + pi_uniform * (3 - p)
            theta_fixed = model.get_fixed_params(gauge="hierarchical", pi_lc=pi_lc)
            thetas[position] = theta_fixed
        thetas = pd.DataFrame(thetas)
        thetas["orbit"] = [x[0] for x in thetas.index]
        thetas["subseq"] = [x[1] for x in thetas.index]

        print("  Aligning local models around each register")
        theta_ps = {}
        orbits = get_subsets_of_set((0, 1, 2, 3, 4, 5))
        features = get_orbits_features(orbits, model.alphabet_list)
        for p, position in enumerate(positions):
            features_p = [
                (tuple(x + p for x in orbit), subseq) for orbit, subseq in features
            ]
            theta_p = thetas.loc[features_p, position]  # type: ignore
            theta_p.index = features
            theta_ps[position] = theta_p
        theta_ps = pd.DataFrame(theta_ps)
        theta_ps["orbit"] = [x[0] for x in theta_ps.index]
        theta_ps["subseq"] = [x[1] for x in theta_ps.index]
        theta_ps["k"] = [len(x) for x in theta_ps["subseq"]]
        theta_ps0 = theta_ps.loc[theta_ps["k"] == 0, :].copy()
        theta_ps0['label'] = motif_label
        theta0.append(theta_ps0)
        
        print("  Saving gauge-fixed parameters")
        theta_ps.to_csv(f"results/theta_fixed.{motif_label}.aligned.csv")
    
    theta0 = pd.concat(theta0, axis=0)
    print("Saving gauge-fixed parameters for k=0")
    theta0.to_csv("results/theta_fixed0.extended_core.csv")
        
    print("Done.")
