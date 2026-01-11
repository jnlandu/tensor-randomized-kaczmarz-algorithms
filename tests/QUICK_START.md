# Testing quick reference for `trk_algorithms`

This is a quick reference guide for running tests in the `trk_algorithms` package. For full details, see [tests/README.md](tests/README.md).

## Setup (One Time)
```bash
pip install -e ".[test]"
# or
bash setup_tests.sh
```

## Run Tests
```bash
pytest                                    # All tests
pytest -v                                 # Verbose
pytest -v --cov=trk_algorithms           # With coverage
pytest tests/test_utils.py               # One file
pytest -k "partition"                    # Match pattern
make test                                # Using Make
```

## Coverage
```bash
pytest --cov=trk_algorithms --cov-report=html
open htmlcov/index.html                  # macOS
```

## Structure of the tests folder
```
tests/
├── test_utils.py      # 50+ tests for utils.py
├── __init__.py
└── README.md          # Full guide
```

## Quick test template
```python
import pytest
from trk_algorithms.utils import your_function

class TestYourFunction:
    def test_basic(self):
        result = your_function(input)
        assert result == expected
    
    def test_error(self):
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

## Common Patterns

### Torch Tensors
```python
assert result.shape == (10, 5, 8)
assert torch.allclose(result, expected, atol=1e-6)
```

### Parametrize
```python
@pytest.mark.parametrize("n,s", [(20, 4), (50, 5)])
def test_cases(self, n, s):
    assert True
```

### Fixtures
```python
@pytest.fixture
def sample_data():
    return torch.randn(10, 5, 8)

def test_with_fixture(self, sample_data):
    assert sample_data.shape == (10, 5, 8)
```

## Files Created
- ✅ tests/test_utils.py (50+ tests)
- ✅ tests/README.md (full guide)  
- ✅ pytest.ini (config)
- ✅ Makefile (shortcuts)
- ✅ .github/workflows/tests.yml (CI/CD)
- ✅ TESTING.md (this summary)
- ✅ Updated pyproject.toml

## Next Steps
1. `pip install -e ".[test]"` - Install
2. `pytest -v` - Run tests
3. See TESTING.md for details
