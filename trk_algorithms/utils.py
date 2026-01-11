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

from tensor_toolbox.config import DTYPE, device as _DEVICE
from tensor_toolbox.tensorLinalg import (
    t_frobenius_norm,
    t_pinv_apply,
    t_product
)

from trk_algorithms.config import SEED



def as_torch_device(dev) -> torch.device:
    """
    Normalize device from external config (str/torch.device) to torch.device.
    """
    if isinstance(dev, torch.device):
        return dev
    return torch.device(dev)

# Normalize device:
_DEFAULT_DEVICE = as_torch_device(_DEVICE)
device = _DEFAULT_DEVICE

_TORCH_GEN_CPU = torch.Generator(device="cpu")
_TORCH_GEN_CPU.manual_seed(int(SEED))

_NP_RNG = np.random.default_rng(int(SEED))

def make_partitions(n, s=None, tau=2, sequential=True, rng=None):
    """
    Create a partitions of the set of indices {0, ..., n-1} into s disjoint subsets.
    If random is True, the partitions are created randomly.
    Otherwise, the partitions are created sequentially, following the paper 
    "Randomized Block Extended Kaczmarz" by Due et al (2024), that is partitions have the same size tau and are formed as
     I_i = {itau,..., min((i+1)tau-1, n)-1} for i=0,...,s-1.
    Parameters:
    n (int): Total number of indices.
    s (int, optional): Number of partitions. If None, it is computed as ceil(n/tau).
    tau (int, optional): Size of each partition when random is False. Default is 2.
    sequential(bool, optional): Whether to create random partitions. Default is True.

    Returns:
    list: A list containing s lists, each representing a partition of indices.

    If s is None and sequential is True, s is computed as ceil(n/tau). 
    And the partitions are created sequentially, as follows:
     I_i  is formed as  I_i = {(i-1)tau +1, (i-1)tau +2, ..., i*tau} for i=1,...,s-1 and I_s = {(s-1)tau +1, ..., n}.
    
    if s is not None and sequential is True, the partitions are created sequentially, as follows:
     I_i  is formed as  I_i = {(i-1)tau +1, (i-1)tau +2, ..., i*tau} for i=1,...,s-1 and I_s = {(s-1)tau +1, ..., n}.

    If sequential is False, the partitions are created randomly, by randomly permuting the indices 
    {0, ..., n-1} and splitting them into s disjoint subsets of (not necessarily equal) size.

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
    assert sequential in [True, False], "sequential must be a boolean value."

    if s is not None:
        assert isinstance(s, int) and s >= 1, "s must be a positive integer."

    #  Numpy implementation
    if sequential:
        if s is None:
            s = int(np.ceil(n / tau))
        else:
            # Validate that we get exactly s nonempty blocks and cover {0, ..., n-1}.
            # Condition: (s-1)*tau < n <= s*tau.
            if s == 1:
                assert tau >= n, "For s=1, need tau >= n to cover all indices."
            else:
                assert (s - 1) * tau < n, "Need (s-1)*tau < n so the last block is nonempty."
                assert s * tau >= n, "Need s*tau >= n so partitions cover all indices."

        #  Create the partitions following Due et al (2024) REBK paper
        #  I_i  is formed as  I_i = {(i-1)tau +1, (i-1)tau +2, ..., i*tau} for i=1,...,s-1 and I_s = {(s-1)tau +1, ..., n}.
        parts = []
        for i in range(s):
            start = i * tau
            end = min( (i+1) * tau, n )
            if start < n:
                part = list(range(start, end))
                parts.append(part)
        return parts
    
    # If sequential is False, create random partitions
    # rng = np.random.default_rng(seed=SEED)
    # indices = rng.permutation(n)
    if s is None:
        # Determine number of partitions based on tau
        s = int(np.ceil(n / tau))
    else:
        assert s >= 1
    rng = _NP_RNG if rng is None else rng
    indices = rng.permutation(n)
    partitions = np.array_split(indices, s)
    return [parts.tolist() for parts in partitions]


def tau_range(n: int, s: int, style: str = "sequential", cap_at_n: bool = True):
    """
    Return the feasible integer range of ``tau`` for a given ``n`` and ``s``.

    This is mainly meant to answer: “for a fixed problem size ``n`` and a desired
    number of partitions ``s``, what values of ``tau`` are compatible with the
    tau-based partitioning logic?”

    For the *sequential* (Due et al. style) blocks used in :func:`make_partitions`,
    feasibility is equivalent to having exactly ``s`` non-empty blocks with
    block size ``tau`` (last block may be smaller), i.e.:

    $$ (s-1)\,\tau < n \le s\,\tau $$

    which gives the inclusive bounds:

    $$ \lceil n/s \rceil \le \tau \le \left\lfloor (n-1)/(s-1) \right\rfloor \quad (s>1) $$

    For the *random* style in :func:`make_partitions` when ``s`` is provided,
    ``tau`` is not used (only ``s`` matters). In that case this function returns
    ``(1, n)`` by default (or ``(1, None)`` if ``cap_at_n=False``).

    Parameters
    ----------
    n:
        Total number of indices (size of the set {0, ..., n-1}).
    s:
        Desired number of partitions.
    style:
        One of {"sequential", "random"}.
    cap_at_n:
        If True, upper bounds are capped at ``n`` when the theoretical upper
        bound is unbounded (e.g. ``s=1`` or random style).

    Returns
    -------
    (tau_min, tau_max):
        Inclusive integer bounds for feasible ``tau``. ``tau_max`` may be None
        if unbounded and ``cap_at_n=False``.

    Raises
    ------
    ValueError
        If inputs are invalid or no feasible ``tau`` exists.

    Examples
    --------
    Sequential (tau controls the number of blocks):
    >>> tau_range(n=80, s=4, style="sequential")
    (20, 26)

    Random style (tau is ignored when s is provided):
    >>> tau_range(n=80, s=4, style="random")
    (1, 80)
    """

    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer")
    if not (isinstance(s, int) and s > 0):
        raise ValueError("s must be a positive integer")

    style_norm = str(style).strip().lower()
    if style_norm not in {"sequential", "random"}:
        raise ValueError('style must be one of {"sequential", "random"}')

    # In random mode (when s is provided), tau is not used by make_partitions.
    if style_norm == "random":
        return (1, n) if cap_at_n else (1, None)

    # Sequential mode: tau must yield exactly s non-empty blocks.
    if s == 1:
        # Any tau >= n yields a single block covering all indices.
        tau_min = n
        tau_max = n if cap_at_n else None
        return tau_min, tau_max

    # For s > 1, require: (s-1)*tau < n <= s*tau
    tau_min = int(np.ceil(n / s))
    tau_max = (n - 1) // (s - 1)
    if cap_at_n:
        tau_max = min(tau_max, n)

    if tau_min > tau_max:
        raise ValueError(
            f"No feasible tau for n={n}, s={s} under sequential style. "
            f"Need ceil(n/s) <= floor((n-1)/(s-1)). Got {tau_min}..{tau_max}."
        )
    return tau_min, tau_max


def partitions_to_torch(parts, device):
    """
    Convert list[list[int]] -> list[torch.LongTensor] on 'device'.
    
    Parameters:
    -----------
    parts: list of list of int. Partitions to convert.
    device: torch.device. Device to use.
    
    """
    return [torch.tensor(I, dtype=torch.long, device=device) for I in parts]


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
def make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, generator=None):
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
    generator: torch.Generator. random number generator
    


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

    if generator is None:
        # generator = _TORCH_GEN_CPU
        g = torch.Generator(device="cpu")
        g.manual_seed(int(SEED))
    else:
        g = generator


    A = torch.randn(m, n, p, device='cpu', dtype=DTYPE, generator=g).to(device)
    X_true = torch.randn(n, q, p, device='cpu', dtype=DTYPE, generator=g).to(device)

    B_cons = t_product(A, X_true)
    # noise_term = torch.randn(m, q, p, device="cpu", dtype=DTYPE, generator=g).to(device)
    B_incons = B_cons + noise * torch.randn(m, q, p, device='cpu', dtype=DTYPE, generator=g).to(device)
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
        
    Examples:
    ---------
    Minimal:
    >>> results = [
    ...     {'name': 'TREK', 'time': t_trek, 'final_residual': hist_trek[-1], 'iterations': k_trek},
    ...     {'name': 'TREGBK', 'time': t_tregrebk, 'final_residual': hist_tregrebk[-1], 'iterations': k_tregrebk},
    ...     {'name': 'TGDBEK _f', 'time': t_tgdbek_f, 'final_residual': hist_tgdbek_f[-1], 'iterations': k_tgdbek_f},
    ... ]
    >>> df = display_results(results)

    Full set (include whichever methods you ran):
    >>> results = [
    ...     {'name': 'TREK', 'time': t_trek, 'final_residual': hist_trek[-1], 'iterations': k_trek},
    ...     {'name': 'TREABK', 'time': t_treabk, 'final_residual': hist_treabk[-1], 'iterations': k_treabk},
    ...     {'name': 'TREB-G', 'time': t_trebg, 'final_residual': hist_trebg[-1], 'iterations': k_trebg},
    ...     {'name': 'TEGBK', 'time': t_tegbk, 'final_residual': hist_tegbk[-1], 'iterations': k_tegbk},
    ...     {'name': 'TREGREBK', 'time': t_tregrebk, 'final_residual': hist_tregrebk[-1], 'iterations': k_tregrebk},
    ...     {'name': 'TGDBEK (Proposed)', 'time': t_tgdbek, 'final_residual': hist_tgdbek[-1], 'iterations': k_tgdbek},
    ...     {'name': 'TGDBEK (faithful)', 'time': t_tgdbek_f, 'final_residual': hist_tgdbek_f[-1], 'iterations': k_tgdbek_f},
    ... ]
    >>> df = display_results(results)
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
        
    Examples:
    ---------
    Minimal:
    >>> histories = [
    ...     {'name': 'TREK', 'history': hist_trek, 'iterations': k_trek, 'marker': 'o'},
    ...     {'name': 'TREGREBK', 'history': hist_tregrebk, 'iterations': k_tregrebk, 'marker': 'd'},
    ...     {'name': 'TGDBEK (faithful)', 'history': hist_tgdbek_f, 'iterations': k_tgdbek_f,
    ...      'linewidth': 2.5, 'linestyle': '--', 'marker': 'd'}
    ... ]
    >>> plot_convergence(histories)

    Full set (include whichever histories you computed):
    >>> histories = [
    ...     {'name': 'TREK', 'history': hist_trek, 'iterations': k_trek, 'marker': 'o'},
    ...     {'name': 'TREABK', 'history': hist_treabk, 'iterations': k_treabk, 'marker': 'o'},
    ...     {'name': 'TREB-G', 'history': hist_trebg, 'iterations': k_trebg, 'marker': 's'},
    ...     {'name': 'TEGBK', 'history': hist_tegbk, 'iterations': k_tegbk, 'marker': '^'},
    ...     {'name': 'TREGREBK', 'history': hist_tregrebk, 'iterations': k_tregrebk, 'marker': 'd'},
    ...     {'name': 'TGDBEK (Proposed)', 'history': hist_tgdbek, 'iterations': k_tgdbek,
    ...      'linewidth': 2.5, 'linestyle': '--', 'marker': 'd'},
    ...     {'name': 'TGDBEK (faithful)', 'history': hist_tgdbek_f, 'iterations': k_tgdbek_f,
    ...      'linewidth': 2.5, 'linestyle': '--', 'marker': 'd'},
    ... ]
    >>> plot_convergence(histories, save_path='tensor_kaczmarz_convergence.png')
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
