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

See `docs/features.md` for more details on feature generation.

## Testing

To run tests:

```bash
# Activate the conda environment
conda activate working

# Run all tests
pytest

# Run specific test file
pytest tests/test_get_features_upto_order.py
```

## Documentation

Documentation is available in the `docs/` directory:

- `docs/features.md`: Documentation on feature generation functions
