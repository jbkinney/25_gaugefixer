# GaugeFixer

Tools for gauge fixing in sequence analysis, enabling robust modeling of biological sequences.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gaugefixer.git
cd gaugefixer

# Install in development mode
pip install -e .
```

## Package Structure

The package is organized into several modules:

- `features`: Tools for generating features for sequence analysis
- `sequence`: Embedding and evaluation tools for sequences
- `fixers`: Implementation of various gauge fixing methods
- `projection_matrices`: Utilities for handling projection matrices
- `verify`: Verification and validation utilities

## Core Functionality

### Feature Generation

Generate features for sequence analysis using:

```python
from gaugefixer.features import get_features_upto_order

# Generate all features up to order 2 for sequences of length 3 with binary alphabet
features = get_features_upto_order(L=3, max_order=2, alphabet=['A', 'B'])
```

### Sequence Embedding

Embed sequences into feature vectors:

```python
from gaugefixer.sequence import SeqEmbedder
from gaugefixer.features import get_features_upto_order

# Generate features
features = get_features_upto_order(L=3, max_order=2, alphabet=['A', 'C', 'G', 'T'])

# Create embedder
embedder = SeqEmbedder(features, L=3)

# Embed a sequence into a binary feature vector
embedding = embedder.embed('ACG')
```

### Gauge Fixing

Fix the gauge of parameters in various interaction models:

```python
from gaugefixer.fixers import fix_pairwise_model
from gaugefixer.features import get_pairwise_features
import pandas as pd
import numpy as np

# First create model parameters
L = 5  # Sequence length
alphabet = ['A', 'C', 'G', 'T']

# Get the appropriate features for a pairwise model
features = get_pairwise_features(L=L, alphabet=alphabet)

# Create random parameters as a Pandas Series indexed by features
np.random.seed(42)  # For reproducibility
random_params = np.random.normal(0, 1, size=len(features))
model_parameters = pd.Series(data=random_params, index=features)

# Fix gauge of pairwise model parameters using zero-sum gauge
fixed_theta = fix_pairwise_model(
    theta=model_parameters,
    gauge='zero-sum',
    L=L,
    alphabet=alphabet
)

# Fix gauge relative to a wild-type sequence
fixed_theta = fix_pairwise_model(
    theta=model_parameters,
    gauge='wild-type',
    L=L,
    alphabet=alphabet,
    wt_seq='ACGTA'
)

# Fix gauge using custom background frequencies
custom_freqs = np.ones((L, len(alphabet))) / len(alphabet)  # Uniform frequencies
fixed_theta = fix_pairwise_model(
    theta=model_parameters,
    gauge=None,
    L=L,
    alphabet=alphabet,
    pi_lc=custom_freqs
)
```

Other available fixers include:
- `fix_additive_model`: For additive (site-independent) models
- `fix_allorder_model`: For models with interactions of all orders

## Requirements

- Python ≥ 3.9
- NumPy
- Pandas
- typeguard

## Testing

Run tests with:

```bash
pytest
```

## License

See the LICENSE file for details.