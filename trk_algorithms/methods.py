# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trk_algorithms/methods.py
Implementation of Tensor Randomized Kaczmarz Algorithms
"""
import time
import torch
import numpy as np


from tensor_toolbox.config import device, DTYPE
from tensor_toolbox.tensorLinalg import (
    t_product,
    t_transpose,
    t_frobenius_norm,
    t_pinv_apply,
    tube_tpinv
)
from trk_algorithms.utils import (
    make_partitions,
    partitions_to_torch,
    rel_se
)

from trk_algorithms.config import SEED



# ---------- Methods from the paper: 
#  Tensor randomized extended Kaczmarz methods for large 
# inconsistent tensor linear equations with t-product. by
# Guang-Xin Hugang, Shuang-Yo Zhong
# Numerical Algorithms (2024) 96:1755–1778
# https://doi.org/10.1007/s11075-023-01684-w


# -----------------------------------------
#  1. Tensor Randomized Extended Kaczmarz
# -----------------------------------------

def trek_algorithm(A, B, T, x_ls, tol=1e-5):
    """
    Tensor randomized extended Kaczmarz algorithm.
    A tensor version of the Randomized Extended Kaczmarz (REK) method.
    It solves  the tensor linear sysrtem A *_t X = B,
    which may be inconsistent.

    Parameters:
    -----------
    A: torch. Tensor of size (m, n, p).

    B: torch. Tensor of size (m, k, p).

    T: int. Number of iterations.

    x_ls: torch. The least-square solution.

    tol: stopping tolerance (relative solution error).


    Returns:
    --------
    (X, k, res_hist, x_hist), runtime  with:
    X: torch. Tensor of size (n, k, p).

    k: int. Number of iterations.

    res_hist: list of relative solution errors.

    x_hist: list of least-square solutions.

    runtime: float. Time taken to run the algorithm.


    Example:
    --------
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> (X, iters, res_hist, x_hist), runtime = trek_algorithm(
    ...     A=torch.tensor(A, dtype=DTYPE, device=device),
    ...     B=torch.tensor(B, dtype=DTYPE, device=device),
    ...     T=1000,
    ...     x_ls=torch.tensor(X_ls, dtype=DTYPE, device=device),
    ...     tol=1e-5,
    ...     pinv_tol=1e-12,
    ...     seed=42
    ... )
    >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
    Converged in 50 iterations, runtime: 1.2345 seconds.
    >>> ## Plot the convergence
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogy(res_hist)
    >>> plt.xlabel('Iteration')
    >>> plt.ylabel('Relative Solution Error (RSE)')
    >>> plt.title('Convergence of TREK Algorithm')
    >>> plt.grid()
    >>> plt.show()
    """
    torch.manual_seed(SEED)

    m, n, p = A.shape
    m_b, k, p_b = B.shape

    assert m == m_b , "First dimensions of A and B must match."
    assert p == p_b, "Third dimensions must match"
    # assert (m == m_b) and (p == p_b), "A:(m, n, p), B:(m, k, p) required."


    # -------- -- Precompute row and column probabilities  ----------
    #  Cp;umn norms: ||A(:,j,:)||_F^2
    col_norms_sq =torch.sum(A ** 2, dim=(0, 2)) + 1e-12   # (n,)
    #  Row norms: ||A(i,:,:)||_F^2
    row_norms_sq =torch.sum(A ** 2, dim=(1, 2)) + 1e-12   # (m,)

    #Total norm: ||A||_F^2
    total_norm = torch.sum(A**2).to(torch.float32) #=col_norms_sq.sum()  # == row_norms_sq.sum()

    # ---------------- compute row and column probabilities  ----------------
    p_col = col_norms_sq / (total_norm + 1e-12)  # (n,)
    p_row = row_norms_sq / (total_norm + 1e-12)  # (m,)
    # ---------------- end compute row and column probabilities  ----------------

    # Initialize: X^0 = 0, Z^0 = B
    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
    Z = B.clone()
    
    res_hist = []
    x_hist = []


    t0 = time.time()
    for iter_k in range(T):

        # =======================================================
        # Z-update (column step)
        # Pick j with prob ||A(:,j,:)||_F^2 / ||A||_F^2
        # =======================================================
        j = int(torch.multinomial(p_col, 1).item())

        # Update z: Z = Z - A(:,j,:) * (A(:,j,:)^* * A(:,j,:))^† * (A(:,j,:)^* * Z)
        A_col = A[:, j:j+1, :]              # (m,1,p)
        trans_A = t_transpose(A_col)        # (1,m,p)  (paper's A_{:,j,:}^*)
        tmp = t_product(trans_A, Z)         # (1,k,p)
        denomZ = t_frobenius_norm(A_col) ** 2  # = col_norms_sq[j] #|| A(:, j, :) ||_F^2
        Z = Z - t_product(A_col, tmp) / (denomZ + 1e-12)   # (m,k,p)

        # =======================================================
        # X-update (row step)
        # Pick i with prob ||A(i,:,:)||_F^2 / ||A||_F^2
        # =======================================================
        i = int(torch.multinomial(p_row, 1).item())

        # Update X: X = X - A_row^* * (A_row * A_row^*)^† * r_i

        A_row = A[i:i+1, :, :]                  # (1,n,p)
        trans_A_row = t_transpose(A_row)        # (n,1,p)  (paper's A_{i,:,:}^*)
        A_row_X = t_product(A_row, X)           # (1,k,p)
        R_I = A_row_X - B[i:i+1, :, :] + Z[i:i+1, :, :]   # (1,k,p)
        denomX = t_frobenius_norm(A_row) ** 2      # = row_norms_sq[i]  # || A(i, :, :) ||_F^2
        X = X - t_product(trans_A_row, R_I) / (denomX + 1e-12)

        # ----- Monitoring  the RSE  ----------
        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())
        x_hist.append(X.clone())

        if rse < tol:
            break

    runtime = time.time() - t0
    X_np = X.detach().cpu().numpy()

    return (X_np, iter_k + 1, np.array(res_hist), np.array([x.cpu().numpy() for x in x_hist])), runtime

# --------------------------------------------------
#  2. Tensor Randomized Block Kaczmarz
# ---------------------------------------------------
def trebk_algorithm(A, B, T, x_ls, row_partitions=None, col_partitions=None,tol=1e-5):
    """
    Tensor Randomized Extended Block Kaczmarz (TREBK) algorithm.

    Parameters
    ----------
    A: (m, n, p) tensor
    B: (m, k, p) tensor
    T: max iterations
    row_partitions: list of LongTensor index blocks (partition of rows). If None: 10 blocks.
    col_partitions: list of LongTensor index blocks (partition of columns). If None: 10 blocks.
    tol: stopping tolerance on relative residual ||B - A*X||_F / ||B||_F


    Returns
    -------
    (X, iters, res_hist, x_hist), runtime, with :
    X: (n, k, p) tensor
    iters: number of iterations
    res_hist: list of relative solution errors
    x_hist: list of least-square solutions

    When row_partitions is None, default to 10 partitions of equal size.
    In this case, the partitions are randomly generated, after a permutation of the row indices.

    Example:
    --------
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> row_partitions = make_partitions(m=120, s=10)
    >>>> col_partitions = make_partitions(n=80, s=10)
    >>> (X, iters, res_hist, x_hist), runtime = trbk_algorithm(A, B, T=1000, x_ls=X_ls,
    ...     row_partitions=row_partitions, col_partitions= col_partitions, tol=1e-5)
    >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
    """


    m, n, p = A.shape
    mb, k, pb = B.shape

    assert m == mb, "First dimensions of A and B must match."
    assert p == pb, "Third dimensions must match"

    # -------- -- Precompute row and column probabilities  ----------
    if row_partitions is None:
        row_partitions = make_partitions(m, s=10, sequential=False)
        # Convertis les partitions en torch.long
        row_partitions = partitions_to_torch(row_partitions, device=device)
    else:
        #  Convertis les indices des partitions en torch.long
        row_partitions = partitions_to_torch(row_partitions, device=device)
    
    if col_partitions is None:
        col_partitions = make_partitions(n, s=10, sequential=False)
        # Convertis les partitions en torch.long
        col_partitions = partitions_to_torch(col_partitions, device=device)
    else:
        #  Convertis les indices des partitions en torch.long
        col_partitions = partitions_to_torch(col_partitions, device=device)

    # --------- -- Precompute row and column norms  ----------
    # Column block norms: ||A(:,J,: )||_F^2
    col_norms_sq =  torch.stack([torch.sum(A[:, J, :] ** 2) for J in col_partitions]).to(A.device)  # (t,)
    
    # Row  block norms: ||A(I,:,: )||_F^2
    row_norms_sq =  torch.stack([torch.sum(A[I, :, :] ** 2) for I in row_partitions]).to(A.device)  # (s,)

    # Total norm: ||A||_F^2
    total_norm = torch.sum(A**2).to(torch.float32)  # =col_norms_sq.sum()  # == row_norms_sq.sum()

    #  ----------------- compute row and column block probabilities  ----------------
    p_col = col_norms_sq / (total_norm + 1e-12)  # (t,)
    p_row = row_norms_sq / (total_norm + 1e-12)  # (s,)
    #  ----------------- end compute row and column block probabilities  ----------------

    # Initialize
    X = torch.zeros(n, k, p, dtype=DTYPE, device=A.device)
    Z = B.clone()

    res_hist = []
    x_hist = []

    t0 = time.time()
    for iter_k in range(T):

        # =======================================================
        # Z-update (column block step)
        # Pick j in [t] with prob ||A(:,J,:)||_F^2 / ||A||_F^2
        # =======================================================
        j = int(torch.multinomial(p_col, 1).item())
        J = col_partitions[j]

        # Update z: Z = Z - A(:,J,:) * (A(:,J,:)^T * A(:,J,:))^† *  Z)
        A_J = A[:, J, :]                     # (m, |J|, p)
        # A_J_T = t_transpose(A_J)             # (|J|, m, p)
        AJZ = t_pinv_apply(A_J, Z)           # (|J|, k, p)
        Z = Z - t_product(A_J, AJZ)          # (m,k,p)

        # =======================================================
        # X-update (row block step)
        # Pick i in [s] with prob ||A(I,:,: )||_F^2 / ||A||_F^2
        # =======================================================
        i = int(torch.multinomial(p_row, 1).item())
        I = row_partitions[i]

        # Update X: X = X - A_I^T * (A_I * A_I^T)^† * r_I
        A_I = A[I, :, :]                      # (|I|, n, p)
        A_I_X = t_product(A_I, X)             # (|I|, k, p)
        R_I = A_I_X - B[I, :, :] + Z[I, :, :] # (|I|, k, p)
        step = t_pinv_apply(A_I, R_I)         # (n, k, p)
        X = X - step

        # ----- Monitoring  the RSE  ----------
        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())
        x_hist.append(X.clone())
        if rse < tol:
            break
    runtime = time.time() - t0
    #  Mets X en numpy
    X_np = X.detach().cpu().numpy()
    return (X_np, iter_k + 1, np.array(res_hist), np.array([x.cpu().numpy() for x in x_hist])), runtime


# -----------------------------------------------------
#  3. Tensor Randomized Extended Greedy Block Kaczmarz
# ----------------------------------------------------
def tregbk_algorithm(A, B, T, x_ls, delta=0.9, row_partitions=None,
                     tol=1e-5, rcond=1e-12, seed=0):
    """
    Tensor randomized extended greedy block Kaczmarz (TREGBK).


    Parameters
    ----------
    A: (m, n, p) tensor
    B: (m, k, p) tensor
    T: max iterations
    delta: greedy parameter  in (0,1]
    row_partitions: list of LongTensor index blocks (partition of rows). If None: 10 blocks.
    tol: stopping tolerance on relative residual ||B - A*X||_F / ||B||_F
    rcond: rcond for slice-wise pinv
    seed: RNG seed
    store_hist: store X at each iteration (can be large)

    Returns
    -------
    (X, Z, iters, res_hist, x_hist), runtime, with :
    X: (n, k, p) tensor
    iters: number of iterations
    res_hist: list of relative solution errors
    x_hist: list of least-square solutions

    When row_partitions is None, default to 10 partitions of equal size.
    In this case, the partitions are randomly generated, after a permutation of the row indices.

    Example:
    --------
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> row_partitions = make_partitions(m=120, s=10)
    >>> (X, iters, res_hist, x_hist), runtime = tregbk_algorithm(
    ...     A=torch.tensor(A, dtype=DTYPE, device=device),
    ...     B=torch.tensor(B, dtype=DTYPE, device=device),
    ...     T=1000,
    ...     x_ls=torch.tensor(X_ls, dtype=DTYPE, device=device),
    ...     delta=0.9,
    ...     row_partitions=row_partitions,
    ...     tol=1e-5,
    ...     rcond=1e-12,
    ...     seed=42
    ... )
    >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
    Converged in 57 iterations, runtime: 3.1234 seconds
    """
    torch.manual_seed(seed)

    m, n, p = A.shape
    mB, k, pB = B.shape
    assert (m == mB) and (p == pB), "Need A:(m,n,p), B:(m,k,p)"
    assert 0.0 < delta <= 1.0, "delta must be in (0,1]"

    # Default deterministic partitions if not provided. Randomly shuffled
    # with  size ~=10
    if row_partitions is None:
        row_partitions = make_partitions(m, s=10, sequential=False)
        # Convertis les partitions en torch.long
        row_partitions = partitions_to_torch(row_partitions, device=device)
    else:
        #  Convertis les indices des partitions en torch.long
        row_partitions = partitions_to_torch(row_partitions, device=device)


    # Row-block probabilities: ||A_{I_i,:,:}||_F^2 / ||A||_F^2
    row_norms_sq = torch.sum(A**2, dim=(1, 2)).to(torch.float32)  # (m,)
    block_norms  = torch.stack([row_norms_sq[I].sum() for I in row_partitions]).to(A.device) # (s,)
    prob_blocks = (block_norms / block_norms.sum()).to(torch.float32 )  # (s,)  

    #  Precompute A^T 
    At = t_transpose(A) # (n,m,p)

    # Initialize
    X = torch.zeros(n, k, p, dtype=DTYPE, device=A.device)
    Z = B.clone()

    res_hist = []
    x_hist = []


    t0 = time.time()
    for iter_k in range(T):

        # =======================================================
        # 1) Greedy column-block selection tau_k using current Z
        #    u_n = || (A_{:,j,:})^T * Z ||_F^2
        # =======================================================

        # I must vectorize this loop.  I noticed an error in this algorithm description in the paper.
        # They ddefinee || A(:,j,:)*Z||_F^2, which is incorrect.
        # Vectorize version of the  this loop
        AtZ  = t_product(At, Z)                    # (n,k,p) = (n,m,p) * (m,k,p)
        Atz_norms_sq = torch.sum(AtZ**2, dim=(1,2)).to(torch.float32)  # (n,)
        eps = delta * torch.max(Atz_norms_sq)
        tau = torch.where(Atz_norms_sq >= eps)[0]
       
        # =======================================================
        # 5) Z update: Z <- Z - A_{:,tau,:} * (A_{:,tau,:})^† * Z
        # =======================================================
        A_tau = A[:, tau, :]                            # (m, |tau|, p)
        A_tau_Z = t_pinv_apply(A_tau, Z, rcond=rcond)   # (|tau|, k, p)
        Z = Z - t_product(A_tau, A_tau_Z)               # (m, k, p)

        # =======================================================
        # 6) Pick a row-block I_i with probability ||A_I||_F^2 / ||A||_F^2
        # =======================================================
        i = int(torch.multinomial(prob_blocks, 1).item())
        I = row_partitions[i]

        # =======================================================
        # 7) X update : X <- X - (A_I)^† * (A_I*X - B_I + Z_I)
        # =======================================================
        A_I = A[I, :, :]                                    # (|I|, n, p)
        A_I_X = t_product(A_I, X)                           # (|I|, k, p)
        R_I = A_I_X - B[I, :, :] + Z[I, :, :]               # (|I|, k, p)
        step = t_pinv_apply(A_I, R_I, rcond=rcond)          # (n, k, p)
        X = X - step

        #  -------- Monitoring  the RSE  ----------
        assert x_ls is not None, "x_ls (least-square solution) must be provided for RSE computation."

        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())

        x_hist.append(X.clone())

        #  Check convergence
        if rse < tol:
            break

    runtime = time.time() - t0

    # Convert final X to CPU numpy array
    X_np = X.detach().cpu().numpy()
    x_hist_np = np.stack([xx.detach().cpu().numpy() for xx in x_hist], axis=0)

    return (X_np, iter_k + 1, np.array(res_hist), x_hist_np), runtime


# ---------- -------------------------
# Method from the paper:
# "Randomized extended average block Kaczmarz method
# for inconsistent tensor equations under t-product", by 
# Liyuan An, Kun Liang, Han Jiao1 Qilong Liu.
# Numerical Algorithms (2025) 100:1123–1144
# https://doi.org/10.1007/s11075-024-01982-x

# -----------------------------------------
#  Tensor Randomized Extended Average Block Kaczmarz
# -----------------------------------------

def treabk_algorithm(A, B, T,  x_ls, row_partitions=None, col_partitions=None, alpha=1.0, tol=1e-5):
    """
    Tensor Randomized Extended Average Block Kaczmarz (TREABK).
    Uses t-product throughout.

    Parameters:
    ----------
    A: (m, n, p) tensor
    B: (m, k, p) tensor
    T: max iterations
    alpha: relaxation parameter
    row_partitions: list of LongTensor index blocks (partition of rows). If None: 2 blocks.
    col_partitions: list of LongTensor index blocks (partition of columns). If None: 2 blocks.
    tol: stopping tolerance
    Return
    ------
    (X, iter_k, res_hist, x_hist), runtime with:
     X: (n, k, p) tensor
    iter_k: number of iterations
    res_hist: residual history
    x_hist: X history
    runtime: float. time taken to run the algorithm.

    If the row_partitions is None, default to 2 partitions of size 2.
    If the col_partitions is None, default to 2 partitions of size 2.
    In this case, the partitions are randomly generated, after a permutation of the row/column indices.

    Example:
    --------
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> row_partitions = make_partitions(m=120, s=2, tau=2)
    >>> col_partitions = make_partitions(n=80, s=2, tau=2)
    >>> (X, iters, res_hist, x_hist), runtime = treabk_algorithm( A,B,T=1000,X_ls,row_partitions=row_partitions,
    ...     col_partitions=col_partitions,
    ...     alpha=1.0,
    ...     tol=1e-5
    ... )
    >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
      Converged in 60 iterations, runtime: 2.3456 seconds
    >>> ## Plot the convergence 
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogy(res_hist)
    >>> plt.xlabel('Iteration')
    >>> plt.ylabel('Relative Solution Error (RSE)')
    >>> plt.title('Convergence of TREABK Algorithm')
    >>> plt.grid()
    >>> plt.show()
    """
    m, n, p = A.shape
    m_b, k, p_b = B.shape
    assert m == m_b, "First dimensions of A and B must match."
    assert p == p_b, "Third dimensions must match"

    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
    Z = B.clone()

    #  --------- partitions  ----------
    if row_partitions is None:
        row_blocks = 2
        I_blocks = make_partitions(m, s=row_blocks, tau=2, sequential= False)
        I_blocks = partitions_to_torch(I_blocks, device=A.device)
    else:
        I_blocks = partitions_to_torch(row_partitions, device=A.device)
    
    if col_partitions is None:
        #  If the col_partitions is not provided, default to 2  random blokcs of size 2
        col_blocks = 2
        J_blocks = make_partitions(n, s=col_blocks, tau=2, sequential= False)
        J_blocks = partitions_to_torch(J_blocks, device=A.device)
    else:
        J_blocks = partitions_to_torch(col_partitions, device=A.device)

    
    #  ---------------- precompute row and column norms  ----------------
    # Column norms: ||A_{:,J,:}||_F^2
    col_norms_sq = torch.zeros(len(J_blocks), dtype=A.dtype, device=A.device)
    #  Row norms: ||A_{I,:,:}||_F^2
    row_norms_sq = torch.zeros(len(I_blocks), dtype=A.dtype, device=A.device)

    # ---------------- block norms computation  ----------------
    #  Column block norms
    col_block_norms = torch.stack([col_norms_sq[J].sum() for J in J_blocks]).to(A.device)  # (t,)
    #  Row block norms
    row_block_norms = torch.stack([row_norms_sq[I].sum() for I in I_blocks]).to(A.device)  # (s,)

    total_norm = (A**2).sum().to(torch.float32) #= col_block_norms.sum()  # == row_block_norms.sum()
    #  ---------------- end precompute row and column norms  ----------------

    # ---------------- compute row and column probabilities  ----------------
    prob_cols = col_block_norms / (total_norm + 1e-12)  # (t,)
    prob_rows = row_block_norms / (total_norm + 1e-12)  # (s,)
    # ---------------- end compute row and column probabilities  ----------------

    #  Initialize row and column norms for blocks
    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)  # (n,k,p) 
    Z = B.clone()                                             # (m,k,p)

    res_hist = []
    x_hist = []

    t0 = time.time()
    for iter_k in range(T):
        # Column step: pick j in [t] with probability prop to ||A_{:,j,:}||_F^2

        j = torch.multinomial(prob_cols, 1).item()
        J = J_blocks[j]
        AJ = A[:, J, :]                       # (m, n, p)

        # Update  Z = Z - alpha * A_col *_t (A_col^T_t * Z) / ||A_col||_F^2
        # A_col_trans = t_transpose(AJ)            # (n, m, p)
        # Atz = t_product(A_col_trans, Z)       
        Atz = t_product(t_transpose(AJ), Z)        # (|J|, k, p)
        AZ = t_product(AJ, Atz)                    # (m, k, p)
        denomZ =( t_frobenius_norm(AJ)**2 + 1e-12)     #=col_norms_sq[j] + 1e-12
        Z = Z - alpha * AZ / denomZ

        # Row step: pick i in [s]  with probability prop to ||A_{i,:,:}||_F^2
        i = torch.multinomial(prob_rows, 1).item()
        I = I_blocks[i]
        A_I = A[I, :, :]                     # ({I|}, n, p)

        # X update: X = X - alpha * A_row^T_t *  r_i / ||A_row||_F^2

        AX_I = t_product(A_I, X)                 # (|I|, k, p)
        RI = AX_I - B[I, :, :] + Z[I, :, :]      # (|I|, k, p)

        # X update: X = X - alpha * A_row^T_t *_t r_i / ||A_row||_F^2
        grad = t_product(t_transpose(A_I), RI)  # (n, k, p)
        denomX = (t_frobenius_norm(A_I)**2 + 1e-12)   # = row_norms_sq[i] + 1e-12

        X = X - alpha * (grad / denomX)

        #  Monitor the RSE: exigere x_ls
        assert x_ls is not None, "x_ls (least-square solution) must be provided for RSE computation."

        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())
        x_hist.append(X.clone())

        #  Convergence test:
        if rse < tol:
            break

    runtime = time.time() - t0

    # Convert final X to CPU numpy array
    X_np = X.detach().cpu().numpy()
    return (X_np, iter_k + 1, np.array(res_hist), np.array([x.cpu().numpy() for x in x_hist])), runtime

# -----------------------------------------
#  End of File
# -----------------------------------------






#  I Must  remove these lines  later

# def treb_greedy_algorithm(A, B, T, x_ls, delta=0.9, tol=1e-5):
#     """
#     Tensor Randomized Extended Greedy Block Kaczmarz (TREBGK).
#     Baseline with greedy selection.

#     Parameters:
#     A: (m, n, p) tensor
#     B: (m, k, p) tensor
#     T: max iterations
#     delta: greedy threshold parameter in (0, 1]
#     tol: stopping tolerance

#     Return
#     ------
#     (X, iter_k, res_hist, x_hist), runtime with:
#      X: (n, k, p) tensor
#     iter_k: number of iterations
#     res_hist: residual history
#     x_hist: X history

#     Example:
#     --------    
#     >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
#     >>> (X, iters, res_hist, x_hist), runtime = treb_greedy_algorithm(
#     ...     A=torch.tensor(A, dtype=DTYPE, device=device),
#     ...     B=torch.tensor(B, dtype=DTYPE, device=device),
#     ...     T=1000,
#     ...     x_ls=torch.tensor(X_ls, dtype=DTYPE, device=device),
#     ...     delta=0.9,
#     ...     tol=1e-5
#     ... )
#     >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
#     Converged in 45 iterations, runtime: 2.5678 seconds 
#     """

#     m, n, p = A.shape
#     m_b, k, p_b = B.shape

#     assert m == m_b , "First dimensions of A and B must match."
#     assert p == p_b, "Third dimensions must match"

#     X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
#     Z = B.clone()

#     col_norms_sq = torch.sum(A ** 2, dim=(0, 2)) + 1e-12
#     row_norms_sq = torch.sum(A ** 2, dim=(1, 2)) + 1e-12

#     res_hist = []
#     x_hist = []

#     t0 = time.time()
#     for iter_k in range(T):
#         # Column selection: compute ||A_{:,j,:}^T_t * Z||_F^2 for all j
#         u_norms = torch.zeros(n, device=A.device)
#         for j in range(n):
#             A_col = A[:, j:j+1, :]  # (m, 1, p)
#             A_col_trans = t_transpose(A_col)
#             u_j = t_product(A_col_trans, Z)  # (1, k, p)
#             u_norms[j] = t_frobenius_norm(u_j) ** 2

#         threshold_u = delta * u_norms.max()
#         U_k = torch.where(u_norms >= threshold_u)[0]

#         if U_k.numel() == 0:
#             U_k = torch.tensor([u_norms.argmax()], device=A.device)

#         # Update Z with selected columns
#         for j in U_k:
#             A_col = A[:, j:j+1, :]
#             A_col_trans = t_transpose(A_col)
#             u_j = t_product(A_col_trans, Z)
#             update = t_product(A_col, u_j)
#             Z = Z - update / col_norms_sq[j]

#         # Row selection: compute ||B - Z - A *_t X||_F^2 for all i
#         R = B - Z - t_product(A, X)
#         r_norms = torch.sum(R ** 2, dim=(1, 2))

#         threshold_r = delta * r_norms.max()
#         J_k = torch.where(r_norms >= threshold_r)[0]

#         if J_k.numel() == 0:
#             J_k = torch.tensor([r_norms.argmax()], device=A.device)

#         # Update X with selected rows
#         for i in J_k:
#             A_row = A[i:i+1, :, :]
#             r_i = R[i:i+1, :, :]
#             A_row_trans = t_transpose(A_row)
#             update_x = t_product(A_row_trans, r_i)
#             X = X + update_x / row_norms_sq[i]

#         rse = rel_se(X, x_ls)
#         res_hist.append(rse.item())

#         x_hist.append(X.clone())

#         if rse < tol:
#             break
#     runtime = time.time() - t0

#     # Convert final X to CPU numpy array
#     X_np = X.detach().cpu().numpy()
#     return (X_np, iter_k + 1, np.array(res_hist), np.array([x.detach().cpu().numpy() for x in x_hist])), runtime
