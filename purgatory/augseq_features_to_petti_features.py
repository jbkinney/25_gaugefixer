def augseq_features_to_petti_features(features):
    N = len(features)
    L = len(features[0])
    
    # For each feature, get positions of non-wildcard characters
    positions = []
    subseqs = []
    for feature in features:
        pos = tuple([i for i, c in enumerate(feature) if c != '*'])
        subseq = feature.replace('*', '')
        positions.append(pos)
        subseqs.append(subseq)
    return list(zip(positions, subseqs))
    
    
    
    
    