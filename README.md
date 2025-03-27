# Gauge Fixing and Augmented Sequence Generator

This repository contains tools for implementing gauge fixing on augmented sequences and generating augmented sequences of various orders, as described in the paper.

## Main Functions

### 1. Gauge Fixing

The `gauge_fix_sequences` function implements gauge fixing according to Eq. 19 (eq:projection_matrix) from the paper. It transforms augmented sequences with wildcards according to the specified gauge.

```python
from src.gauge_fixing import gauge_fix_sequences
import pandas as pd

# Example usage
df = pd.DataFrame({
    'sequence': ['A*C', 'AB*', '*BC'],
    'theta': [0.1, 0.2, 0.3]
})

# Probability matrix as DataFrame
p_lc = pd.DataFrame({
    'A': [0.7, 0.3, 0.0],
    'B': [0.2, 0.6, 0.1],
    'C': [0.0, 0.0, 0.8]
}, index=[0, 1, 2])

# Apply gauge fixing with lambda = 1.0
fixed_df = gauge_fix_sequences(df, lambda_param=1.0, p_lc=p_lc)
```

### 2. Augmented Sequence Generation

The library provides three levels of augmented sequence generation:

#### a. Generate Sequences in a Single Orbit

```python
from src.get_augseqs_in_orbit import get_augseqs_in_orbit

# Generate all sequences with 'A' or 'B' at positions 0 and 2, and wildcards elsewhere
seqs = get_augseqs_in_orbit(['A', 'B'], [0, 2], 3)
# Returns: ['A*A', 'A*B', 'B*A', 'B*B']
```

#### b. Generate Sequences of a Specific Order

```python
from src.get_augseqs_of_order import get_augseqs_of_order

# Generate all sequences with characters at exactly 2 positions in a length-3 sequence
seqs = get_augseqs_of_order(3, 2, ['A', 'B'])
# Returns sequences with characters at all possible combinations of 2 positions
```

#### c. Generate Sequences up to a Maximum Order

```python
from src.get_augseqs_upto_order import get_augseqs_upto_order

# Generate all sequences with characters at up to 2 positions in a length-3 sequence
seqs = get_augseqs_upto_order(3, 2, ['A', 'B'])
# Returns sequences from order 0 (all wildcards) up to order 2
```

## Testing

Tests are implemented using pytest and located in the `tests/` directory. To run the tests:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_gauge_fixing.py
```

## Dependencies

- numpy
- pandas
- pytest (for testing) 