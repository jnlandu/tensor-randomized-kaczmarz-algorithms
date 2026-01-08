# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trk_algorithms/methods.py
Implementation of Tensor Randomized Kaczmarz Algorithms
"""

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
    rel_se
)

from trk_algorithms.config import SEED
import time


# -----------------------------------------
#  Tensor Randomized Extended Kaczmarz
# -----------------------------------------

def trek_algorithm(A, B, T, x_ls, tol=1e-5, pinv_tol=1e-12, seed=SEED):
    """
    Tensor randomized extended Kaczmarz algorithm.

    Parameters:
    -----------
    A: torch. Tensor of size (m, n, p).

    B: torch. Tensor of size (m, k, p).

    T: int. Number of iterations.

    x_ls: torch. The least-square solution.

    tol: stopping tolerance (relative solution error).

    pinv_tol: tolerance used inside tube pseudoinverse.

    seed: random seed. 

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
    """
    torch.manual_seed(seed)

    m, n, p = A.shape
    m_b, k, p_b = B.shape

    assert (m == m_b) and (p == p_b), "A:(m, n, p), B:(m, k, p) required."

    # Initialize
    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
    Z = B.clone()

    absA2 = A * A

    col_norms_sq = absA2.sum(dim=(0, 2)) + 1e-12   # (n,)
    row_norms_sq = absA2.sum(dim=(1, 2)) + 1e-12   # (m,)
    p_col = col_norms_sq / col_norms_sq.sum()
    p_row = row_norms_sq / row_norms_sq.sum()

    res_hist = []
    x_hist = []

    denomB = t_frobenius_norm(B) + 1e-12

    t0 = time.time()

    for iter_k in range(T):

        # =======================================================
        # Z-update (column step)
        # Pick j with prob ||A(:,j,:)||_F^2 / ||A||_F^2
        # Z <- Z - A(:,j,:) * (A(:,j,:)^* * A(:,j,:))^† * A(:,j,:)^* * Z
        # =======================================================
        j = int(torch.multinomial(p_col, 1).item())

        A_col = A[:, j:j+1, :]           # (m,1,p)
        # (1,m,p)  (this is the paper's A_{:,j,:}^*)
        A_col_T = t_transpose(A_col)

        # G = A_col^* * A_col is (1,1,p)
        G = t_product(A_col_T, A_col)    # (1,1,p)
        Gdag = tube_tpinv(G, tol=pinv_tol)

        u = t_product(A_col_T, Z)        # (1,k,p)
        u = t_product(Gdag, u)           # (1,k,p)
        Z = Z - t_product(A_col, u)      # (m,k,p)

        # =======================================================
        # X-update (row step)
        # Pick i with prob ||A(i,:,:)||_F^2 / ||A||_F^2
        # X <- X - A(i,:,:)^* * (A(i,:,:) * A(i,:,:)^*)^† * (A(i,:,:) * X - B(i,:,:) + Z(i,:,:))
        # =======================================================
        i = int(torch.multinomial(p_row, 1).item())

        A_row = A[i:i+1, :, :]           # (1,n,p)
        A_row_T = t_transpose(A_row)     # (n,1,p)  (paper's A_{i,:,:}^*)

        H = t_product(A_row, A_row_T)    # (1,1,p)  = A_row * A_row^*
        Hdag = tube_tpinv(H, tol=pinv_tol)

        resid_i = t_product(A_row, X) - \
            B[i:i+1, :, :] + Z[i:i+1, :, :]   # (1,k,p)
        step = t_product(A_row_T, t_product(
            Hdag, resid_i))              # (n,k,p)
        X = X - step

        # =======================================================
        # Convergence check (relative residual)
        # =======================================================

        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())
        x_hist.append(X.clone())

        if rse < tol:
            break

    runtime = time.time() - t0
    X_np = X.detach().cpu().numpy()

    return (X_np, iter_k + 1, np.array(res_hist), np.array([x.cpu().numpy() for x in x_hist])), runtime


# -----------------------------------------
#  Tensor Randomized Extended Average Block Kaczmarz
# -----------------------------------------

def treabk_algorithm(A, B, T,  x_ls, row_blocks=10, col_blocks=10, alpha=1.0, tol=1e-5):
    """
    Tensor Randomized Extended Average Block Kaczmarz (TREABK).
    Uses t-product throughout.

    Parameters:
    ----------
    A: (m, n, p) tensor
    B: (m, k, p) tensor
    T: max iterations
    alpha: relaxation parameter
    s: number of blocks for partitioning
    tol: stopping tolerance
    Return
    ------
    (X, iter_k, res_hist, x_hist), runtime with:
     X: (n, k, p) tensor
    iter_k: number of iterations
    res_hist: residual history
    x_hist: X history
    runtime: float. time taken to run the algorithm.

    Example:
    --------
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> (X, iters, res_hist, x_hist), runtime = treabk_algorithm(
    ...     A=torch.tensor(A, dtype=DTYPE, device=device),
    ...     B=torch.tensor(B, dtype=DTYPE, device=device),
    ...     T=1000,
    ...     x_ls=torch.tensor(X_ls, dtype=DTYPE, device=device),
    ...     row_blocks=10,
    ...     col_blocks=10,
    ...     alpha=1.0,
    ...     tol=1e-5
    ... )
    >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
    Converged in 60 iterations, runtime: 2.3456 seconds
    """
    m, n, p = A.shape
    m_b, k, p_b = B.shape
    assert p == p_b, "Third dimensions must match"

    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
    Z = B.clone()

    #  Partitions:
    I_blocks = torch.tensor_split(torch.arange(m, device=A.device), row_blocks)
    J_blocks = torch.tensor_split(torch.arange(n, device=A.device), col_blocks)

    # Compute row norms: ||A_{I,:,:}||_F^2
    row_norms_sq_list = []
    for I in I_blocks:
        row_norms_sq_list.append((A[I, :, :] * A[I, :, :]).sum())
    row_norms_sq = torch.stack(row_norms_sq_list).to(A.device) + 1e-12

    # Compute column norms: ||A_{:,J,:}||_F^2
    col_norms_sq_list = []
    for J in J_blocks:
        col_norms_sq_list.append((A[:, J, :] * A[:, J, :]).sum())
    col_norms_sq = torch.stack(col_norms_sq_list).to(A.device) + 1e-12

    # Column probabilities:
    prob_cols = col_norms_sq / col_norms_sq.sum()

    # Row probabilities
    prob_rows = row_norms_sq / row_norms_sq.sum()

    res_hist = []
    x_hist = []

    t0 = time.time()
    for iter_k in range(T):
        # Column step: pick j with probability prop to ||A_{:,j,:}||_F^2

        j = torch.multinomial(prob_cols, 1).item()
        J = J_blocks[j]
        AJ = A[:, J, :]  # (m, n, p)
        A_col_trans = t_transpose(AJ)  # (n, m, p)
        Atz = t_product(A_col_trans, Z)  # (n, k, p)

        # denomJ = frob(AJ) + 1e-12

        # Z-update
        # Z update: Z = Z - alpha * A_col *_t (A_col^T_t *_t Z) / ||A_col||_F^2
        update = t_product(AJ, Atz)  # (m, k, p)
        Z = Z - alpha * update / col_norms_sq[j]

        # Row step: pick i with probability prop to ||A_{i,:,:}||_F^2
        prob_rows = row_norms_sq / row_norms_sq.sum()
        i = torch.multinomial(prob_rows, 1).item()
        I = I_blocks[i]
        A_row = A[I, :, :]  # (m, n, p)

        # Compute residual: A_row *_t X - B_i
        AX = t_product(A_row, X)  # (m, k, p)
        RI = AX - B[I, :, :] + Z[I, :, :]  # (m, n, p)

        # X update: X = X - alpha * A_row^T_t *_t r_i / ||A_row||_F^2
        A_row_trans = t_transpose(A_row)  # (n, m, p)
        X = X - alpha * t_product(A_row_trans, RI) / row_norms_sq[i]

        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())

        x_hist.append(X.clone())

        if rse < tol:
            break

    runtime = time.time() - t0

    # Convert final X to CPU numpy array
    X_np = X.detach().cpu().numpy()
    return (X_np, iter_k + 1, np.array(res_hist), np.array([x.cpu().numpy() for x in x_hist])), runtime


# --------------------------------------------------
#  Tensor Randomized Extended Greedy Block Kaczmarz
# ---------------------------------------------------

def treb_greedy_algorithm(A, B, T, x_ls, delta=0.9, tol=1e-5):
    """
    Tensor Randomized Extended Greedy Block Kaczmarz (TREBGK).
    Baseline with greedy selection.

    Parameters:
    A: (m, n, p) tensor
    B: (m, k, p) tensor
    T: max iterations
    delta: greedy threshold parameter in (0, 1]
    tol: stopping tolerance

    Return
    ------
    (X, iter_k, res_hist, x_hist), runtime with:
     X: (n, k, p) tensor
    iter_k: number of iterations
    res_hist: residual history
    x_hist: X history

    Example:
    --------    
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> (X, iters, res_hist, x_hist), runtime = treb_greedy_algorithm(
    ...     A=torch.tensor(A, dtype=DTYPE, device=device),
    ...     B=torch.tensor(B, dtype=DTYPE, device=device),
    ...     T=1000,
    ...     x_ls=torch.tensor(X_ls, dtype=DTYPE, device=device),
    ...     delta=0.9,
    ...     tol=1e-5
    ... )
    >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
    Converged in 45 iterations, runtime: 2.5678 seconds 
    """

    _, n, p = A.shape
    _, k, p_b = B.shape

    assert p == p_b, "Third dimensions must match"

    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
    Z = B.clone()

    col_norms_sq = torch.sum(A ** 2, dim=(0, 2)) + 1e-12
    row_norms_sq = torch.sum(A ** 2, dim=(1, 2)) + 1e-12

    res_hist = []
    x_hist = []

    t0 = time.time()
    for iter_k in range(T):
        # Column selection: compute ||A_{:,j,:}^T_t * Z||_F^2 for all j
        u_norms = torch.zeros(n, device=A.device)
        for j in range(n):
            A_col = A[:, j:j+1, :]  # (m, 1, p)
            A_col_trans = t_transpose(A_col)
            u_j = t_product(A_col_trans, Z)  # (1, k, p)
            u_norms[j] = t_frobenius_norm(u_j) ** 2

        threshold_u = delta * u_norms.max()
        U_k = torch.where(u_norms >= threshold_u)[0]

        if U_k.numel() == 0:
            U_k = torch.tensor([u_norms.argmax()], device=A.device)

        # Update Z with selected columns
        for j in U_k:
            A_col = A[:, j:j+1, :]
            A_col_trans = t_transpose(A_col)
            u_j = t_product(A_col_trans, Z)
            update = t_product(A_col, u_j)
            Z = Z - update / col_norms_sq[j]

        # Row selection: compute ||B - Z - A *_t X||_F^2 for all i
        R = B - Z - t_product(A, X)
        r_norms = torch.sum(R ** 2, dim=(1, 2))

        threshold_r = delta * r_norms.max()
        J_k = torch.where(r_norms >= threshold_r)[0]

        if J_k.numel() == 0:
            J_k = torch.tensor([r_norms.argmax()], device=A.device)

        # Update X with selected rows
        for i in J_k:
            A_row = A[i:i+1, :, :]
            r_i = R[i:i+1, :, :]
            A_row_trans = t_transpose(A_row)
            update_x = t_product(A_row_trans, r_i)
            X = X + update_x / row_norms_sq[i]

        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())

        x_hist.append(X.clone())

        if rse < tol:
            break
    runtime = time.time() - t0

    # Convert final X to CPU numpy array
    X_np = X.detach().cpu().numpy()
    return (X_np, iter_k + 1, np.array(res_hist), np.array([x.detach().cpu().numpy() for x in x_hist])), runtime


# -----------------------------------------
#  Tensor Randomized Extended Greedy Block Kaczmarz
# -----------------------------------------
def tregbk_algorithm(A, B, T, x_ls, delta=0.9, row_partitions=None,
                     tol=1e-5, rcond=1e-12, seed=0):
    """
    Algorithm 3: Tensor randomized extended greedy block Kaczmarz (TREGREBK).

    Require:
      A ∈ R^{mxnxp}, B ∈ R^{mxkxp}, partitions {I1,...,Is} of [m], δ ∈ (0,1]

    Steps each iter:
      - Greedy column-block tau_k via energies || (A_{:,j,:})^T * Z ||_F^2
      - Z <- Z - A_{:,tau,:} * (A_{:,tau,:})^† * Z
      - Pick a row-block I_i with probability ||A_{I_i,:,:}||_F^2 / ||A||_F^2
      - X <- X - (A_{I_i,:,:})^† * (A_{I_i,:,:}*X - B_{I_i,:,:} + Z_{I_i,:,:})

    Parameters
    ----------
    A: (m, n, p) tensor
    B: (m, k, p) tensor
    T: max iterations
    delta: greedy parameter δ in (0,1]
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

    # Default deterministic partitions if not provided
    if row_partitions is None:
        row_partitions = make_partitions(m, s=min(10, m))

    # Initialize
    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
    Z = B.clone()

    # Precompute column tube norms for sampling/normalization if needed
    # if torch.is_complex(A):
    #     absA2 = (A.conj() * A).real
    # else:
    absA2 = A * A

    # Row-block probabilities: ||A_{I_i,:,:}||_F^2 / ||A||_F^2
    A_frob2 = absA2.sum() + 1e-12
    block_norm2_list = []
    for I in row_partitions:
        block_norm2_list.append(absA2[I, :, :].sum() + 1e-12)
    block_norm2 = torch.stack(block_norm2_list).to(A.device)
    p_blocks = (block_norm2 / block_norm2.sum()).to(torch.float32)

    res_hist = []
    x_hist = []
    denomB = torch.linalg.norm(B.reshape(-1)) + 1e-12

    t0 = time.time()
    for iter_k in range(T):

        # =======================================================
        # 1) Greedy column-block selection tau_k using current Z
        #    energy_j = || (A_{:,j,:})^T * Z ||_F^2
        # =======================================================
        energies = torch.zeros(n, device=A.device, dtype=torch.float32)

        for j in range(n):
            A_col = A[:, j:j+1, :]                 # (m,1,p)
            # (1,m,p)  (your t_transpose)
            A_col_T = t_transpose(A_col)
            u_j = t_product(A_col_T, Z)            # (1,k,p)
            # squared Frobenius norm
            energies[j] = (t_frobenius_norm(u_j) ** 2).to(torch.float32)

        thr = delta * energies.max()
        tau = torch.where(energies >= thr)[0]
        if tau.numel() == 0:
            tau = torch.tensor(
                [int(torch.argmax(energies).item())], device=A.device)

        # =======================================================
        # 2) Z update: Z <- Z - A_{:,tau,:} * (A_{:,tau,:})^† * Z
        # =======================================================
        A_tau = A[:, tau, :]                        # (m, |tau|, p)
        X_tau = t_pinv_apply(A_tau, Z, rcond=rcond)  # (|tau|, k, p)
        Z = Z - t_product(A_tau, X_tau)             # (m, k, p)

        # =======================================================
        # 3) Pick a row-block I_i with probability ||A_I||_F^2 / ||A||_F^2
        # =======================================================
        blk_id = int(torch.multinomial(p_blocks, 1).item())
        # I = row_partitions[blk_id].to(A.device)
        I = row_partitions[blk_id]

        # =======================================================
        # 4) X update (extended Kaczmarz):
        #    X <- X - (A_I)^† * (A_I*X - B_I + Z_I)
        # =======================================================
        A_I = A[I, :, :]                              # (|I|, n, p)
        rhs = t_product(A_I, X) - B[I, :, :] + Z[I, :, :]   # (|I|, k, p)
        step = t_pinv_apply(A_I, rhs, rcond=rcond)          # (n, k, p)
        X = X - step

        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())

        x_hist.append(X.clone())

        if rse < tol:
            break

    runtime = time.time() - t0

    # Convert final X to CPU numpy array
    X_np = X.detach().cpu().numpy()
    x_hist_np = np.stack([xx.detach().cpu().numpy() for xx in x_hist], axis=0)

    return (X_np, iter_k + 1, np.array(res_hist), x_hist_np), runtime


def tegbk_algorithm(A, B, T, x_ls, alpha=1.0, delta=0.9, tol=1e-5):
    """
    Tensor Extended Greedy Block Kaczmarz (TEGBK), 

    Parameters:
    A: (m, n, p) tensor
    B: (m, k, p) tensor
    T: max iterations
    alpha: relaxation parameter
    delta: greedy threshold parameter in (0, 1]
    tol: stopping tolerance
    Return
    ------
    (X, iter_k, res_hist, x_hist), runtime with:
     X: (n, k, p) tensor
    iter_k: number of iterations
    res_hist: residual history
    x_hist: X history
    runtime: float. time taken to run the algorithm.
    Example:
    --------    
    >>> A, X_ls, B = make_tensor_problem(m=120, n=80, p=8, q=4, noise=0.05, seed=42)
    >>> (X, iters, res_hist, x_hist), runtime = tegbk_algorithm(
    ...     A=torch.tensor(A, dtype=DTYPE, device=device),
    ...     B=torch.tensor(B, dtype=DTYPE, device=device),  
    ...     T=1000,
    ...     x_ls=torch.tensor(X_ls, dtype=DTYPE, device=device),
    ...     alpha=1.0,
    ...     delta=0.9,
    ...     tol=1e-5
    ... )
    >>> print(f"Converged in {iters} iterations, runtime: {runtime:.4f} seconds")
    Converged in 52 iterations, runtime: 1.987  
    """
    m, n, p = A.shape
    m_b, k, p_b = B.shape

    X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
    Z = B.clone()

    col_norms_sq = torch.sum(A ** 2, dim=(0, 2)) + 1e-12
    row_norms_sq = torch.sum(A ** 2, dim=(1, 2)) + 1e-12

    res_hist = []
    x_hist = []

    t0 = time.time()
    for iter_k in range(T):
        # Column step with greedy selection
        u_norms = torch.zeros(n, device=A.device)
        for j in range(n):
            A_col = A[:, j:j+1, :]
            A_col_trans = t_transpose(A_col)
            u_j = t_product(A_col_trans, Z)
            u_norms[j] = t_frobenius_norm(u_j) ** 2

        eps_z = delta * u_norms.max()
        U_k = torch.where(u_norms >= eps_z)[0]

        if U_k.numel() == 0:
            print("U_K is zero")
            U_k = torch.tensor([u_norms.argmax()], device=A.device)

        # Pseudoinverse update for Z
        for j in U_k:
            A_col = A[:, j:j+1, :]
            A_col_trans = t_transpose(A_col)
            u_j = t_product(A_col_trans, Z)
            update = t_product(A_col, u_j)
            Z = Z - update / col_norms_sq[j]
        # AUk = A[:, U_k, :]
        # Auk_trans = t_transpose(AUk)
        # u= t_product(Auk_trans, Z)
        # update = t_product(AUk, u)
        # Z = Z - alpha * update / col_norms_sq[U_k]

        # Z = Z - t_product(AUk, t_product(t_transpose(AUk), Z)) / col_norms_sq[U_k]

        # Row step with greedy selection
        R = B - Z - t_product(A, X)
        r_norms = torch.sum(R ** 2, dim=(1, 2))

        threshold_r = delta * r_norms.max()
        J_k = torch.where(r_norms >= threshold_r)[0]

        if J_k.numel() == 0:
            print("J_k is zero")
            J_k = torch.tensor([r_norms.argmax()], device=A.device)

        # Pseudoinverse update for X
        for i in J_k:
            A_row = A[i:i+1, :, :]
            r_i = R[i:i+1, :, :]
            A_row_trans = t_transpose(A_row)
            update_x = t_product(A_row_trans, r_i)
            X = X + update_x / row_norms_sq[i]
        # ARow = A[J_k, :, :]
        # r_i = R[J_k, :, :]
        # A_row_trans = t_transpose(ARow)
        # update_x = t_product(A_row_trans, r_i)
        # X = X + alpha * update_x / row_norms_sq[J_k]

        # X = X + t_product(ARow, t_product(t_transpose(ARow), X)) / row_norms_sq[J_k]

        rse = rel_se(X, x_ls)
        res_hist.append(rse.item())

        x_hist.append(X.clone())

        if rse < tol:
            break
    runtime = time.time() - t0

    # Convert final X to CPU numpy array
    X_np = X.detach().cpu().numpy()
    return (X_np, iter_k + 1, np.array(res_hist), np.array([x.detach().cpu().numpy() for x in x_hist])), runtime


# -----------------------------------------
#  End of File
# -----------------------------------------
