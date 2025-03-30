def sort_features(features, wildcard_char='*'):
    return sorted(features, key=lambda sp: (len(sp)-sp.count(wildcard_char), sp[::-1]))

def sort_petti_features(petti_features):
    return sorted(petti_features, key=lambda pf: (len(pf[0]), pf[0], pf[1]))
