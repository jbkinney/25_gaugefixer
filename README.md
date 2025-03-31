# GaugeFixer

Tools for gauge fixing in sequence analysis.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gaugefixer.git
cd gaugefixer

# Install in development mode
pip install -e .
```

## Usage

See the `docs/` directory for detailed documentation on specific functionality.

### Feature Generation

The package provides tools to generate features for sequence analysis:

```python
from src.get_features_upto_order import get_features_upto_order

# Generate all features up to order 2 for sequences of length 3 with binary alphabet
features = get_features_upto_order(L=3, max_order=2, alphabet=['A', 'B'])
```

### Sequence Embedding

The package includes functionality to embed sequences into feature vectors:

```python
from src.seq_embedder import SeqEmbedder
from src.get_features_upto_order import get_features_upto_order

# Generate features
features = get_features_upto_order(L=3, max_order=2, alphabet=['A', 'C', 'G', 'T'])

# Create embedder
embedder = SeqEmbedder(features, L=3)

# For better performance with large feature sets, you can disable validation
# embedder = SeqEmbedder(features, L=3, check_features=False)

# Embed a sequence into a binary feature vector
embedding = embedder.embed('ACG')
```

See `docs/features.md` for more details on feature generation and sequence embedding.

## Testing

To run tests:

```bash
# Activate the conda environment
conda activate working

# Run all tests
pytest

# Run specific test file
pytest tests/test_seq_embedder.py
```

## Documentation

Documentation is available in the `docs/` directory:

- `docs/features.md`: Documentation on feature generation and sequence embedding functions
