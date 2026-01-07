#  The utils.py contains the utilities functions such as :
#  - rel_se: compute the relatve squared error between two tensors, mainly the iterate X_k and the least-square solution X_ls
#  - make_partitions: create random partitions of the set of indices {0, ..., n-1} into p disjoint subsets. This method is mainly used for
#    the block  averaging variants of the randomized Kaczmarz algorithms.


#  The tensor toolbox is used for tensor operations under the t-product framework such as t-product, transpose, identity tensor, f-diagonal tensor, etc.
#  The imports above imports automatically numpy and pytorch as well.

import torch
import numpy as np

from tensor_toolbox.CONFIG import dtype, device
from tensor_toolbox.tensorLinalg import (
    t_frobenius_norm,
    t_pinv_apply,
    t_product
)


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
    """
    assert n > 0 and isinstance(n, int), "n must be a positive integer."
    assert tau > 0 and isinstance(tau, int), "tau must be a positive integer."
    assert s is None or (s > 0 and isinstance(
        s, int)), "s must be a positive integer or None."
    assert tau <= n, "tau must be less than or equal to n."
    assert s is None or tau * \
        s >= n, "s is too small to cover all indices with partitions of size tau."
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
    t_frobenius_norm(X - X_ref) / (t_frobenius_norm(X_ref) + 1e-12)
    """

    diff = X - X_ref
    frob_diff = t_frobenius_norm(diff)
    frob_ref = t_frobenius_norm(X_ref)
    rse = frob_diff / (frob_ref + 1e-12)
    return rse


# ------------------------------------------------------
#  Make tensor problems for testing
# ------------------------------------------------------
def make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=1, dtype=dtype, device=device):
    """
    Build an inconsistent t-linear system:
      B = A * X_true + noise

    Parameters:
    ----------
    m: int
    n: int
    p: int
    q: int
    noise: float
    seed: int
    dtype: torch.dtype
    device: torch.device

    Returns:
    --------
    A: (m, n, p) tensor
    X_ls: (n, q, p) tensor
    B: (m, q, p) tensor
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
