"""
Unit tests for trk_algorithms.utils module.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from trk_algorithms.utils import (
    as_torch_device,
    make_partitions,
    tau_range,
    partitions_to_torch,
    rel_se,
    make_tensor_problem,
    display_results,
    plot_convergence
)


class TestAsTorchDevice:
    """Tests for as_torch_device function."""
    
    def test_device_from_string(self):
        """Test converting string to torch.device."""
        dev = as_torch_device("cpu")
        assert isinstance(dev, torch.device)
        assert dev.type == "cpu"
    
    def test_device_from_device(self):
        """Test passing through an existing torch.device."""
        original = torch.device("cpu")
        dev = as_torch_device(original)
        assert isinstance(dev, torch.device)
        assert dev == original
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device conversion."""
        dev = as_torch_device("cuda")
        assert dev.type == "cuda"


class TestMakePartitions:
    """Tests for make_partitions function."""
    
    def test_sequential_basic(self):
        """Test basic sequential partitioning."""
        partitions = make_partitions(n=20, s=4, tau=5, sequential=True)
        assert len(partitions) == 4
        assert partitions[0] == [0, 1, 2, 3, 4]
        assert partitions[1] == [5, 6, 7, 8, 9]
        assert partitions[2] == [10, 11, 12, 13, 14]
        assert partitions[3] == [15, 16, 17, 18, 19]
    
    def test_sequential_auto_s(self):
        """
        Test sequential partitioning with automatic s calculation.
        """
        partitions = make_partitions(n=23, s=None, tau=6, sequential=True)
        assert len(partitions) == 4
        # Last partition should contain remaining elements
        assert partitions[-1] == [18, 19, 20, 21, 22]
    
    def test_sequential_unequal_last(self):
        """Test sequential with last partition smaller."""
        partitions = make_partitions(n=22, s=4, tau=6, sequential=True)
        assert len(partitions) == 4
        assert len(partitions[0]) == 6
        assert len(partitions[-1]) == 4  # Last partition is smaller
    
    def test_random_partitions(self):
        """Test random partitioning."""
        partitions = make_partitions(n=20, s=4, sequential=False)
        assert len(partitions) == 4
        # All elements should be present
        all_indices = sorted([idx for part in partitions for idx in part])
        assert all_indices == list(range(20))
    
    def test_single_partition(self):
        """Test single partition case."""
        partitions = make_partitions(n=10, s=1, tau=10, sequential=True)
        assert len(partitions) == 1
        assert partitions[0] == list(range(10))
    
    def test_invalid_n(self):
        """Test invalid n parameter."""
        with pytest.raises(AssertionError):
            make_partitions(n=-5, s=2, tau=2)
    
    def test_invalid_tau(self):
        """Test invalid tau parameter."""
        with pytest.raises(AssertionError):
            make_partitions(n=10, s=2, tau=-1)
    
    def test_tau_too_large(self):
        """Test tau larger than n."""
        with pytest.raises(AssertionError):
            make_partitions(n=10, s=2, tau=15)
    
    def test_coverage(self):
        """Test that all indices are covered exactly once."""
        partitions = make_partitions(n=50, s=7, tau=8, sequential=True)
        all_indices = [idx for part in partitions for idx in part]
        assert sorted(all_indices) == list(range(50))
        assert len(all_indices) == 50  # No duplicates


class TestTauRange:
    """Tests for tau_range function."""
    
    def test_sequential_basic(self):
        """Test basic sequential tau range calculation."""
        tau_min, tau_max = tau_range(n=80, s=4, style="sequential")
        assert tau_min == 20  # ceil(80/4)
        assert tau_max == 26  # floor(79/3)
    
    def test_sequential_s_equals_1(self):
        """Test tau range when s=1."""
        tau_min, tau_max = tau_range(n=50, s=1, style="sequential")
        assert tau_min == 50
        assert tau_max == 50  # capped at n
    
    def test_sequential_s_equals_1_uncapped(self):
        """Test tau range when s=1 without capping."""
        tau_min, tau_max = tau_range(n=50, s=1, style="sequential", cap_at_n=False)
        assert tau_min == 50
        assert tau_max is None  # unbounded
    
    def test_random_style(self):
        """Test tau range for random style."""
        tau_min, tau_max = tau_range(n=80, s=4, style="random")
        assert tau_min == 1
        assert tau_max == 80
    
    def test_random_style_uncapped(self):
        """Test tau range for random style without capping."""
        tau_min, tau_max = tau_range(n=80, s=4, style="random", cap_at_n=False)
        assert tau_min == 1
        assert tau_max is None
    
    def test_invalid_n(self):
        """Test invalid n parameter."""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            tau_range(n=-5, s=2)
    
    def test_invalid_s(self):
        """Test invalid s parameter."""
        with pytest.raises(ValueError, match="s must be a positive integer"):
            tau_range(n=10, s=0)
    
    def test_invalid_style(self):
        """Test invalid style parameter."""
        with pytest.raises(ValueError, match='style must be one of'):
            tau_range(n=10, s=2, style="invalid")
    
    def test_no_feasible_tau(self):
        """Test case where no feasible tau exists."""
        # For very large s relative to n
        with pytest.raises(ValueError, match="No feasible tau"):
            tau_range(n=10, s=50, style="sequential")
    
    def test_consistency_with_make_partitions(self):
        """Test that tau values in range work with make_partitions."""
        n, s = 80, 4
        tau_min, tau_max = tau_range(n=n, s=s, style="sequential")
        
        # Test min tau
        partitions = make_partitions(n=n, s=s, tau=tau_min, sequential=True)
        assert len(partitions) == s
        
        # Test max tau
        partitions = make_partitions(n=n, s=s, tau=tau_max, sequential=True)
        assert len(partitions) == s


class TestPartitionsToTorch:
    """Tests for partitions_to_torch function."""
    
    def test_basic_conversion(self):
        """Test basic conversion to torch tensors."""
        parts = [[0, 1, 2], [3, 4, 5], [6, 7]]
        torch_parts = partitions_to_torch(parts, device="cpu")
        
        assert len(torch_parts) == 3
        assert all(isinstance(p, torch.Tensor) for p in torch_parts)
        assert torch_parts[0].dtype == torch.long
        assert torch_parts[0].tolist() == [0, 1, 2]
    
    def test_empty_partition(self):
        """Test handling of empty partitions."""
        parts = [[0, 1], [], [2, 3]]
        torch_parts = partitions_to_torch(parts, device="cpu")
        assert len(torch_parts) == 3
        assert len(torch_parts[1]) == 0


class TestRelSe:
    """Tests for rel_se function."""
    
    def test_identical_tensors(self):
        """Test relative error between identical tensors."""
        X = torch.randn(10, 5, 8)
        rse = rel_se(X, X)
        assert rse.item() < 1e-10  # Should be approximately zero
    
    def test_orthogonal_tensors(self):
        """Test relative error between different tensors."""
        X1 = torch.ones(10, 5, 8)
        X2 = torch.zeros(10, 5, 8)
        rse = rel_se(X1, X2)
        assert rse.item() > 0  # Should be non-zero
    
    def test_scaled_tensor(self):
        """Test relative error with scaled tensor."""
        X_ref = torch.randn(10, 5, 8)
        X = 2 * X_ref  # Double the reference
        rse = rel_se(X, X_ref)
        assert rse.item() > 0.9  # Should be approximately 1
        assert rse.item() < 1.1


class TestMakeTensorProblem:
    """Tests for make_tensor_problem function."""
    
    def test_default_parameters(self):
        """Test tensor problem generation with default parameters."""
        A, X_ls, B = make_tensor_problem()
        
        assert A.shape == (120, 80, 8)
        assert X_ls.shape == (80, 4, 8)
        assert B.shape == (120, 4, 8)
    
    def test_custom_dimensions(self):
        """Test with custom dimensions."""
        m, n, p, q = 50, 30, 16, 3
        A, X_ls, B = make_tensor_problem(m=m, n=n, p=p, q=q)
        
        assert A.shape == (m, n, p)
        assert X_ls.shape == (n, q, p)
        assert B.shape == (m, q, p)
    
    def test_reproducibility(self):
        """Test that same generator produces same results."""
        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(42)
        A1, X_ls1, B1 = make_tensor_problem(generator=gen1)
        
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(42)
        A2, X_ls2, B2 = make_tensor_problem(generator=gen2)
        
        assert torch.allclose(A1, A2)
        assert torch.allclose(X_ls1, X_ls2)
        assert torch.allclose(B1, B2)
    
    def test_noise_effect(self):
        """Test that noise affects the problem."""
        A1, X_ls1, B1 = make_tensor_problem(noise=0.0)
        A2, X_ls2, B2 = make_tensor_problem(noise=0.1)
        
        # Different noise should produce different B
        assert not torch.allclose(B1, B2)


class TestDisplayResults:
    """Tests for display_results function."""
    
    def test_basic_display(self, capsys):
        """Test basic results display."""
        results = [
            {'name': 'Method1', 'time': 1.5, 'final_residual': 0.01, 'iterations': 100},
            {'name': 'Method2', 'time': 2.0, 'final_residual': 0.02, 'iterations': 150},
        ]
        
        df = display_results(results)
        
        # Check returned dataframe
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['Method', 'Time (s)', 'Final Relative Residual', 'Iterations']
        
        # Check output was printed
        captured = capsys.readouterr()
        assert "BENCHMARK RESULTS" in captured.out
        assert "Method1" in captured.out
        assert "Method2" in captured.out
    
    def test_empty_results(self):
        """Test with empty results list."""
        results = []
        df = display_results(results)
        assert len(df) == 0
    
    def test_single_method(self):
        """Test with single method."""
        results = [
            {'name': 'TREK', 'time': 1.0, 'final_residual': 0.001, 'iterations': 50},
        ]
        df = display_results(results)
        assert len(df) == 1
        assert df.iloc[0]['Method'] == 'TREK'


class TestPlotConvergence:
    """Tests for plot_convergence function."""
    
    def test_basic_plot(self, tmp_path):
        """Test basic convergence plotting."""
        histories = [
            {'name': 'Method1', 'history': [1.0, 0.5, 0.1, 0.01], 'iterations': 4},
            {'name': 'Method2', 'history': [1.0, 0.6, 0.2, 0.05], 'iterations': 4},
        ]
        
        # Save to temporary file
        save_path = tmp_path / "test_plot.png"
        
        # This will display and save the plot
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for testing
            plot_convergence(histories, save_path=str(save_path))
            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_custom_styling(self, tmp_path):
        """Test plot with custom styling."""
        histories = [
            {
                'name': 'Custom',
                'history': [1.0, 0.5, 0.1],
                'iterations': 3,
                'linewidth': 3.0,
                'linestyle': '--',
                'marker': 's',
                'markersize': 5
            },
        ]
        
        save_path = tmp_path / "custom_plot.png"
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            plot_convergence(histories, save_path=str(save_path))
            assert save_path.exists()
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_empty_histories(self):
        """Test with empty histories list."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            plot_convergence([])  # Should not crash
        except ImportError:
            pytest.skip("matplotlib not available")


# Integration tests
class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self):
        """Test a complete workflow from problem generation to partitioning."""
        # Generate problem
        A, X_ls, B = make_tensor_problem(m=50, n=30, p=8, q=4)
        
        # Create partitions
        n = A.shape[0]
        partitions = make_partitions(n=n, s=5, tau=10, sequential=True)
        
        # Convert to torch
        torch_parts = partitions_to_torch(partitions, device="cpu")
        
        # Verify
        assert len(torch_parts) == 5
        assert all(isinstance(p, torch.Tensor) for p in torch_parts)
    
    def test_tau_range_with_partitions(self):
        """Test that tau_range values work with make_partitions."""
        n, s = 100, 5
        tau_min, tau_max = tau_range(n=n, s=s, style="sequential")
        
        # Test several tau values in range
        for tau in range(tau_min, tau_max + 1):
            partitions = make_partitions(n=n, s=s, tau=tau, sequential=True)
            assert len(partitions) == s
            # Verify all indices covered
            all_indices = [idx for part in partitions for idx in part]
            assert sorted(all_indices) == list(range(n))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
