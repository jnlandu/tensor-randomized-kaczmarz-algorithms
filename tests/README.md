# Testing Guide for trk_algorithms

This document explains how to run and write tests for the trk_algorithms package.

## Setup

### Install Testing Dependencies

First, install pytest and related packages:

```bash
pip install pytest pytest-cov matplotlib
```

Or add to your `requirements.txt`:
```
pytest>=7.0.0
pytest-cov>=4.0.0
matplotlib>=3.5.0
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_utils.py
```

### Run specific test class
```bash
pytest tests/test_utils.py::TestMakePartitions
```

### Run specific test method
```bash
pytest tests/test_utils.py::TestMakePartitions::test_sequential_basic
```

### Run with coverage report
```bash
pytest --cov=trk_algorithms --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run tests matching a pattern
```bash
# Run only tests with "partition" in the name
pytest -k partition

# Run only tests with "tau" in the name
pytest -k tau
```

### Skip slow tests
```bash
pytest -m "not slow"
```

## Test Structure

```
tests/
├── __init__.py           # Makes tests a package
├── test_utils.py         # Tests for utils.py
├── test_methods.py       # Tests for methods.py (to be created)
└── test_config.py        # Tests for config.py (to be created)
```

## Writing New Tests

### Basic Test Structure

```python
import pytest
from trk_algorithms.utils import your_function

class TestYourFunction:
    """Tests for your_function."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = your_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case."""
        result = your_function(edge_input)
        assert result is not None
    
    def test_invalid_input(self):
        """Test that invalid input raises error."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

### Testing with PyTorch Tensors

```python
import torch

def test_tensor_output(self):
    """Test function that returns tensors."""
    result = your_function()
    
    # Check shape
    assert result.shape == (10, 5, 8)
    
    # Check values approximately equal
    expected = torch.zeros(10, 5, 8)
    assert torch.allclose(result, expected, atol=1e-6)
```

### Using Fixtures

```python
@pytest.fixture
def sample_tensor():
    """Fixture providing a sample tensor."""
    return torch.randn(50, 30, 8)

def test_with_fixture(self, sample_tensor):
    """Test using fixture."""
    result = process_tensor(sample_tensor)
    assert result.shape == sample_tensor.shape
```

### Parametrized Tests

```python
@pytest.mark.parametrize("n,s,tau", [
    (20, 4, 5),
    (50, 5, 10),
    (100, 10, 10),
])
def test_multiple_cases(self, n, s, tau):
    """Test multiple parameter combinations."""
    result = make_partitions(n=n, s=s, tau=tau)
    assert len(result) == s
```

### Testing Plots

```python
def test_plot_function(self, tmp_path):
    """Test plot generation."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    save_path = tmp_path / "test_plot.png"
    plot_function(data, save_path=str(save_path))
    
    assert save_path.exists()
```

## Test Categories

### Unit Tests
Test individual functions in isolation:
```python
class TestMakePartitions:
    def test_sequential_basic(self):
        partitions = make_partitions(n=20, s=4, tau=5)
        assert len(partitions) == 4
```

### Integration Tests
Test multiple components working together:
```python
class TestIntegration:
    def test_full_workflow(self):
        A, X_ls, B = make_tensor_problem()
        partitions = make_partitions(n=A.shape[0], s=5)
        torch_parts = partitions_to_torch(partitions, device="cpu")
        assert len(torch_parts) == 5
```

## Continuous Integration

### GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=trk_algorithms --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```


## Example: Adding Tests for New Function

When you add a new function like `new_algorithm()`:

```python
# tests/test_utils.py

class TestNewAlgorithm:
    """Tests for new_algorithm function."""
    
    def test_basic_functionality(self):
        """Test that algorithm works with valid input."""
        result = new_algorithm(input_data)
        assert result is not None
        assert result.shape == expected_shape
    
    def test_convergence(self):
        """Test that algorithm converges."""
        history = new_algorithm(data, return_history=True)
        assert history[-1] < history[0]  # Error decreases
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="dimensions"):
            new_algorithm(wrong_shape_data)
    
    @pytest.mark.slow
    def test_large_problem(self):
        """Test with large problem size."""
        large_data = generate_large_problem()
        result = new_algorithm(large_data)
        assert result.shape == large_data.shape
```

## Troubleshooting

### Import Errors
If you get import errors, install the package in development mode:
```bash
pip install -e .
```

### CUDA Tests
To skip CUDA tests when CUDA is not available:
```python
@pytest.mark.skipif(not torch.cuda.is_available(), 
                    reason="CUDA not available")
def test_cuda_function(self):
    ...
```

### Matplotlib in Tests
Use non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [PyTorch Testing](https://pytorch.org/docs/stable/testing.html)
