def sort_features(features, wildcard_char='*'):
    return sorted(features, key=lambda sp: (len(sp)-sp.count(wildcard_char), sp[::-1]))