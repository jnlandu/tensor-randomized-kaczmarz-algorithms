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
    partitions_to_torch,
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
    Z = B.clone()  # (m,k,p)

    # Compute row norms: ||A_{I,:,:}||_F^2
    # row_norms_sq_list = []
    # for I in I_blocks:
    #     row_norms_sq_list.append((A[I, :, :] * A[I, :, :]).sum())
    # row_norms_sq = torch.stack(row_norms_sq_list).to(A.device) + 1e-12

    # # Compute column norms: ||A_{:,J,:}||_F^2
    # col_norms_sq_list = []
    # for J in J_blocks:
    #     col_norms_sq_list.append((A[:, J, :] * A[:, J, :]).sum())
    # col_norms_sq = torch.stack(col_norms_sq_list).to(A.device) + 1e-12

    # # Column probabilities:
    # prob_cols = col_norms_sq / col_norms_sq.sum()

    # # Row probabilities
    # prob_rows = row_norms_sq / row_norms_sq.sum()

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
      - Greedy column-block tau_k via u_norms || (A_{:,j,:})^T * Z ||_F^2
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
    block_norm2_list = []
    #  ----- A supprimer 
    absA2 = A * A
    for I in row_partitions:
        block_norm2_list.append(absA2[I, :, :].sum() + 1e-12)
    block_norm2 = torch.stack(block_norm2_list).to(A.device)
    p_blocks = (block_norm2 / block_norm2.sum()).to(torch.float32)
    #  -----

    # Initialize
    X = torch.zeros(n, k, p, dtype=DTYPE, device=A.device)
    Z = B.clone()

    res_hist = []
    x_hist = []


    t0 = time.time()
    for iter_k in range(T):

        # =======================================================
        # 1) Greedy column-block selection tau_k using current Z
        #    u_norms[j] = || (A_{:,j,:})^T * Z ||_F^2
        # =======================================================
        u_norms = torch.zeros(n, device=A.device, dtype=torch.float32)

        # I must vectorize this loop. 
        # Vectorize version of the  this loop
        trans_A = t_transpose(A)  # (n,m,p)
        trans_AZ = t_product(trans_A, Z)  # (n,k,p) =  (n, m, p) * (m, k, p) = (n, k, p)
        u_norms = (t_frobenius_norm(trans_AZ, dim=(1, 2)) ** 2).to(torch.float32)  # (n,)
        
        eps_z = delta * torch.max(u_norms)
        tau = torch.where(u_norms >= eps_z)[0]


        # for j in range(n):
        #     A_col = A[:, j:j+1, :]                 # (m,1,p)
        #     A_col_T = t_transpose(A_col)
        #     u_j = t_product(A_col_T, Z)            # (1,k,p)
        #     # squared Frobenius norm
        #     u_norms[j] = (t_frobenius_norm(u_j) ** 2).to(torch.float32)

        # thr = delta * u_norms.max()
        # tau = torch.where(u_norms >= thr)[0]
        # if tau.numel() == 0:
        #     tau = torch.tensor(
        #         [int(torch.argmax(u_norms).item())], device=A.device)

        # =======================================================
        # 5) Z update: Z <- Z - A_{:,tau,:} * (A_{:,tau,:})^† * Z
        # =======================================================
        A_tau = A[:, tau, :]                            # (m, |tau|, p)
        A_tau_Z = t_pinv_apply(A_tau, Z, rcond=rcond)   # (|tau|, k, p)
        Z = Z - t_product(A_tau, A_tau_Z)               # (m, k, p)

        # =======================================================
        # 6) Pick a row-block I_i with probability ||A_I||_F^2 / ||A||_F^2
        # =======================================================
        row_block_idx = int(torch.multinomial(prob_blocks, 1).item())
        # I = row_partitions[blk_id].to(A.device)
        I = row_partitions[row_block_idx]

        # =======================================================
        # 7) X update (extended Kaczmarz):
        #    X <- X - (A_I)^† * (A_I*X - B_I + Z_I)
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
    _, n, p = A.shape
    _, k, p_b = B.shape

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
