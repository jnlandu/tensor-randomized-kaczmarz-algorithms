"""
Docstring for trk_algorithms.utils
"""

#  The utils.py contains the utilities functions such as :
#  - rel_se: compute the relatve squared error between two tensors,
#   mainly the iterate X_k and the least-square solution X_ls
#  - make_partitions: create random partitions of the set of indices {0, ..., n-1} 
# into p disjoint subsets. This method is mainly used for
#    the block  averaging variants of the randomized Kaczmarz algorithms.


#  The tensor toolbox is used for tensor operations under the t-product 
# framework such as t-product, transpose, identity tensor, f-diagonal tensor, etc.
#  The imports above imports automatically numpy and pytorch as well.

import torch
import numpy as np

from tensor_toolbox.config import DTYPE, device
from tensor_toolbox.tensorLinalg import (
    t_frobenius_norm,
    t_pinv_apply,
    t_product
)

from trk_algorithms.config import SEED


def make_partitions(n, s=None, tau=10, sequential=True):
    """
    Create a partitions of the set of indices {0, ..., n-1} into s disjoint subsets.
    If random is True, the partitions are created randomly.
    Otherwise, the partitions are created sequentially, following the paper 
    "Randomized Block Extended Kaczmarz" by Due et al (2024), that is partitions have the same size tau and are formed as
     I_i  is formed as  I_i = {(i-1)tau +1, (i-1)tau +2, ..., i*tau} for i=1,...,s-1 and I_s = {(s-1)tau +1, ..., n}.

    Parameters:
    n (int): Total number of indices.
    s (int, optional): Number of partitions. If None, it is computed as ceil(n/tau).
    tau (int, optional): Size of each partition when random is False. Default is 2.
    sequential(bool, optional): Whether to create random partitions. Default is True.

    Returns:
    list: A list containing s lists, each representing a partition of indices.

    Example:
    --------
    >>> partitions = make_partitions(n=20, s=4, tau=5, sequential=True)
    >>> print(partitions)
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    >>> partitions = make_partitions(n=20, s=4, tau=5, sequential=False)
    >>> print(partitions)
    [array([12,  1,  7, 19,  3,  5,  0, 15,  9, 14]), array([ 4, 11, 17, 10, 13]), array([ 6, 16,  2, 18]), array([8])]
    >>> partitions = make_partitions(n=23, s=None, tau=6, sequential=True)
    >>> print(partitions)
    [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22]]
    """
    assert n > 0 and isinstance(n, int), "n must be a positive integer."
    assert tau > 0 and isinstance(tau, int), "tau must be a positive integer."
    assert s is None or (s > 0 and isinstance(
        s, int)), "s must be a positive integer or None."
    assert tau <= n, "tau must be less than or equal to n."
    # assert s is None or tau * \
    #     s >= n, "s is too small to cover all indices with partitions of size tau."
    assert sequential in [True, False], "sequential must be a boolean value."

    #  Numpy implementation
    if s is None and sequential:
        s = int(np.ceil(n / tau))

    #  Create the partitions following Due et al (2024) REBK paper
    #  I_i  is formed as  I_i = {(i-1)tau +1, (i-1)tau +2, ..., i*tau} for i=1,...,s-1 and I_s = {(s-1)tau +1, ..., n}.
        partitions = [[(i - 1)*tau + j for j in range(1, tau + 1)]
                      for i in range(1, s)]
        return partitions

    if sequential:
        partitions = [[(i - 1)*tau + j for j in range(1, tau + 1)]
                      for i in range(1, s)]

        return partitions
    else:
        # Randonmly generate s partitions of {0, ..., n-1}of not (necessarily) equal size

        indices = np.random.permutation(n)
        #  Partitiion the indices into s disjoint subsets of  not (necessarily) equal size
        partitions = np.array_split(indices, s)

    return partitions


def rel_se(X, X_ref):
    """
    Compute relative solution error.

    Paramaters:
    -----------
    X:  torch.tensor. 3d tensor of shape (n, k, p) 
    X_ref: (n, k, p): reference tensor or leas-squre solution

    Returns:
    --------
    rse: float. relative solution error.

    Example:
    --------
    >>> X = torch.randn(80, 4, 8)
    >>> X_ref = torch.randn(80, 4, 8)
    >>> rse = rel_se(X, X_ref)
    >>> print(f"Relative solution error: {rse:.6f}")
    Relative solution error: 1.234567

    """

    diff = X - X_ref
    frob_diff = t_frobenius_norm(diff)
    frob_ref = t_frobenius_norm(X_ref)
    rse = frob_diff / (frob_ref + 1e-12)
    return rse


# ------------------------------------------------------
#  Make tensor problems for testing
# ------------------------------------------------------
def make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=SEED, dtype=DTYPE, device=device):
    """
    Build an inconsistent t-linear system:
      B = A * X_true + noise

    Parameters:
    ----------
    m: int. number of rows
    n: int. number of columns
    p: int. tubal dimension
    q: int. number of right-hand sides
    noise: float. noise level
    seed: int. random seed
    dtype: torch.dtype. data type
    device: torch.device. device to use

    Returns:
    --------
    A: (m, n, p) tensor.  coefficient tensor
    X_ls: (n, q, p) tensor. least-square solution
    B: (m, q, p) tensor. right-hand side tensor (inconsistent)

    Example:
    --------
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> A.shape
    torch.Size([120, 80, 8])
    >>> X_ls.shape
    torch.Size([80, 4, 8])
    >>> B.shape
    torch.Size([120, 4, 8])
    """

    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    A = torch.randn(m, n, p, device=device, dtype=dtype)
    X_true = torch.randn(n, q, p, device=device, dtype=dtype)

    B_cons = t_product(A, X_true)
    B_incons = B_cons + noise * \
        torch.randn(m, q, p, device=device, dtype=dtype)
    X_ls = t_pinv_apply(A, B_incons)

    return A, X_ls, B_incons


# ------------------------------------------------------
#  Display benchmark results
# ------------------------------------------------------
def display_results(method_results):
    """
    Create and display a summary DataFrame of benchmark results for tensor Kaczmarz methods.
    
    Parameters:
    -----------
    method_results: list of dict
        Each dict should contain:
        - 'name': str, method name
        - 'time': float, execution time in seconds
        - 'final_residual': float, final relative residual
        - 'iterations': int, number of iterations
        
    Example:
    --------
    >>> results = [
    ...     {'name': 'TREK', 'time': t_trek, 'final_residual': hist_trek[-1], 'iterations': k_trek},
    ...     {'name': 'TREABK', 'time': t_treabk, 'final_residual': hist_treabk[-1], 'iterations': k_treabk},
    ... ]
    >>> display_benchmark_results(results)
    """
    import pandas as pd
    
    # Extract data from method_results
    methods = [result['name'] for result in method_results]
    times = [result['time'] for result in method_results]
    residuals = [result['final_residual'] for result in method_results]
    iterations = [result['iterations'] for result in method_results]
    
    # Create DataFrame
    results = pd.DataFrame({
        'Method': methods,
        'Time (s)': times,
        'Final Relative Residual': residuals,
        'Iterations': iterations
    })
    
    # Display results
    print("=" * 90)
    print("BENCHMARK RESULTS - TENSOR KACZMARZ METHODS (Using T-Product)")
    print("=" * 90)
    print(results.to_string(index=False))
    print("=" * 90)
    
    return results


# ------------------------------------------------------
#  Plot convergence results
# ------------------------------------------------------
def plot_convergence(method_histories, figsize=(11, 7), save_path=None):
    """
    Plot convergence histories for tensor Kaczmarz methods.
    
    Parameters:
    -----------
    method_histories: list of dict
        Each dict should contain:
        - 'name': str, method name for legend
        - 'history': array-like, convergence history (e.g., relative residual errors)
        - 'iterations': int, number of iterations
        - 'linewidth': float, optional (default: 2)
        - 'linestyle': str, optional (default: '-')
        - 'marker': str, optional (default: 'o')
        - 'markersize': float, optional (default: 3)
    figsize: tuple, optional
        Figure size (width, height) in inches. Default is (11, 7).
    save_path: str, optional
        If provided, save the figure to this path (e.g., 'convergence.png').
        
    Example:
    --------
    >>> histories = [
    ...     {'name': 'TREK', 'history': hist_trek, 'iterations': k_trek},
    ...     {'name': 'TREGREBK', 'history': hist_tregrebk, 'iterations': k_tregrebk, 'marker': 'd'},
    ...     {'name': 'TGDBEK (faithful)', 'history': hist_tgdbek_f, 'iterations': k_tgdbek_f, 
    ...      'linewidth': 2.5, 'linestyle': '--', 'marker': 'd'}
    ... ]
    >>> plot_convergence(histories)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    
    for method in method_histories:
        name = method['name']
        history = method['history']
        iters = method['iterations']
        linewidth = method.get('linewidth', 2)
        linestyle = method.get('linestyle', '-')
        marker = method.get('marker', 'o')
        markersize = method.get('markersize', 3)
        
        # Calculate marker spacing
        markevery = max(1, len(history) // 20)
        
        # Plot convergence history
        plt.semilogy(history, label=f'{name} - {iters} iters', 
                    linewidth=linewidth, linestyle=linestyle,
                    marker=marker, markersize=markersize, markevery=markevery)
    
    plt.xlabel('IT', fontsize=13)
    plt.ylabel('RSE', fontsize=13)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
