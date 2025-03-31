# Feature Generation Functions

This document describes the feature generation functionality provided by the `gaugefixer` package.

## Overview

The package provides functions to generate features for sequences of a given length. These features are used in sequence analysis and machine learning applications.

A feature is defined as a tuple of:
- A tuple of positions (integers)
- A string representing the subsequence at those positions

For example, the feature `((0, 2), 'AB')` represents the subsequence 'AB' at positions 0 and 2.

## Functions

### `get_features_of_order`

Generates all possible features of a specific order for sequences of a given length.

```python
from src.get_features_of_order import get_features_of_order

# Generate all order-1 features for sequences of length 3 with alphabet ['A', 'B']
features = get_features_of_order(L=3, order=1, alphabet=['A', 'B'])
```

Order-0 features are special case, returning `[((), '')]`, which represents an empty subsequence at no positions.

### `get_features_upto_order`

Generates all possible features from order 0 up to a maximum order for sequences of a given length.

```python
from src.get_features_upto_order import get_features_upto_order

# Generate all features up to order 2 for sequences of length 3 with alphabet ['A', 'B']
features = get_features_upto_order(L=3, max_order=2, alphabet=['A', 'B'])
```

This function uses `get_features_of_order` internally to generate features for each order and combines them.

## Sequence Embedding

The package also provides functionality to embed sequences into a feature space using the generated features.

### `SeqEmbedder`

The `SeqEmbedder` class creates binary feature vectors for sequences by matching patterns derived from the feature set.

```python
from src.seq_embedder import SeqEmbedder
from src.get_features_upto_order import get_features_upto_order

# Generate features
L = 4
alphabet = ['A', 'C', 'G', 'T']
features = get_features_upto_order(L, max_order=2, alphabet=alphabet)

# Create embedder
embedder = SeqEmbedder(features, L)

# Embed a sequence
seq = 'ACGT'
embedding = embedder.embed(seq)  # Returns binary numpy array
```

#### Performance Optimization

For large feature sets, you can use the `check_features` parameter to skip feature validation and improve initialization performance:

```python
# Without validation (faster for large feature sets)
embedder = SeqEmbedder(features, L, check_features=False)
```

Note that disabling validation with `check_features=False` assumes that all features are correctly formatted. Using invalid features without validation may lead to unexpected behavior or runtime errors.

#### How It Works

The `SeqEmbedder` class works as follows:

1. During initialization, it converts each feature into a regular expression pattern:
   - For a feature like `((0, 2), 'AC')` with `L=4`, it creates a pattern like `'A.C.'`
   - The pattern will match any sequence that has 'A' at position 0 and 'C' at position 2

2. When embedding a sequence, it checks if each pattern matches the sequence:
   - If a pattern matches, the corresponding feature value is 1
   - If a pattern doesn't match, the value is 0

3. The result is a binary vector where each element corresponds to a feature.

#### Use Cases

Sequence embedding is useful for:

- Converting sequences into numerical representations for machine learning models
- Comparing sequences based on feature similarities
- Analyzing which features are present in a sequence

## Examples

### Basic Example

```python
from src.get_features_upto_order import get_features_upto_order

# Generate features up to order 1 for a sequence of length 2 with binary alphabet
features = get_features_upto_order(L=2, max_order=1, alphabet=['A', 'B'])

# Result:
# [
#     ((), ''),     # Order 0 feature
#     ((0,), 'A'),  # Order 1: Position 0, character A
#     ((0,), 'B'),  # Order 1: Position 0, character B
#     ((1,), 'A'),  # Order 1: Position 1, character A
#     ((1,), 'B'),  # Order 1: Position 1, character B
# ]
```

### Embedding Example

```python
from src.seq_embedder import SeqEmbedder
from src.get_features_upto_order import get_features_upto_order

# Generate features up to order 1
L = 3
alphabet = ['A', 'C', 'G', 'T']
features = get_features_upto_order(L, max_order=1, alphabet=alphabet)

# Create embedder
embedder = SeqEmbedder(features, L)

# Embed sequences
seq1 = 'ACG'
seq2 = 'TGC'

embedding1 = embedder.embed(seq1)
embedding2 = embedder.embed(seq2)

# Compare embeddings
print(f"Number of features: {len(embedding1)}")
print(f"Number of matching features in seq1: {sum(embedding1)}")
print(f"Number of matching features in seq2: {sum(embedding2)}")
```

### Feature Counts

For a sequence of length `L` with an alphabet of size `|A|`:

- Order 0: 1 feature (empty subsequence)
- Order 1: `L * |A|` features
- Order 2: `(L choose 2) * |A|^2` features
- Order k: `(L choose k) * |A|^k` features

For `get_features_upto_order`, the total number of features is the sum of features from order 0 to `max_order`.

## Running Tests

To run tests for these functions:

```bash
# Activate the conda environment
conda activate working

# Run the tests
pytest tests/test_get_features_of_order.py tests/test_get_features_upto_order.py tests/test_seq_embedder.py
```

Or use the provided script:

```bash
./run_tests.sh
``` 