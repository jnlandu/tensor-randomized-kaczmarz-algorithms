#  Non vectorized version of TGDBEK faithful algorithm
#  This version is kept for reference only.
# def tgdbek_algorithm_faithful(A, B, T, x_ls, eta=0.9, tol=1e-5, rcond=1e-12):
#     """
#     Tensor greedy double block extended Kaczmarz:
#       - greedy sets U_k, J_k
#       - block updates using t-pseudoinverse:
#           Z <- Z - A_U * (A_U)^dagger * Z
#           X <- X + (A_J)^dagger * (B_J - Z_J - A_J*X)

#     Parameters:
#     -----------
#     A: (N1, N2, N3) tensor
#     B: (N1, K, N3) tensor
#     T: max iterations
#     eta: greedy threshold parameter in (0, 1]
#     tol: stopping tolerance

#     Return:
#     -------
#     (X, iter_k, res_hist, x_hist)
#     runtime
#     """
#     m, n, p = A.shape
#     m_b, k, p_b = B.shape
#     assert m == m_b, "First dimensions must match"
#     assert p == p_b, "Third dimensions must match"


#     X = torch.zeros(n, k, p, dtype=A.dtype, device=A.device)
#     Z = B.clone()

#     # norms used only for greedy scoring (as in the screenshot)
#     col_norms_sq = torch.sum(A ** 2, dim=(0, 2)) + 1e-12  # j=0..n-1
#     row_norms_sq = torch.sum(A ** 2, dim=(1, 2)) + 1e-12  # i=0..m-1

#     res_hist = []
#     x_hist = []

#     t0 = time.time()
#     with torch.no_grad():
#         for iter_k in range(T):

#             # -----------------------------
#             # Z-step: build U_k greedily
#             # eps_z^k = eta * max_j ||A(:,j)^T * Z||_F^2 / ||A(:,j)||_F^2
#             # -----------------------------
#             scores_z = torch.zeros(n, device=A.device, dtype=A.dtype)

#             # for j in range(n):
#             #     A_col = A[:, j:j+1, :]                 # (m, 1, p)
#             #     u_j = t_product(t_transpose(A_col), Z) # (1, k, p)
#             #     scores_z[j] = (torch.sum(u_j ** 2)) / col_norms_sq[j]

#             for j in range(n):
#               A_col = A[:, j, :]           # (m, p)
#               A_col_t = A_col.unsqueeze(1) # (m, 1, p)
#               A_col_t_trans = t_transpose(A_col_t)   #(1, m, p)
#               u_j = t_product(A_col_t_trans, Z) ## (1, m, p) * (m, k, p) = (1, k, p)
#               scores_z[j] = (torch.sum(u_j ** 2)) / col_norms_sq[j]

#             # eps_z = eta * torch.max(scores_z)
#             eps_z = eta * scores_z.max()
#             #  Let's the index that gives the max of scores_z
#             j = torch.argmax(scores_z)
#             U_k = torch.where(scores_z >= eps_z)[0]
#             #  Check if  j in UK
#             # if j in U_k:
#             #     print("Ok")
#             # else:
#             #   print("Not ok")


#             # print("Number of ekements of Uk:" , U_k.size())
            
#             # Faithful block update:
#             # Z^{k+1} = Z^k - A_U * (A_U)^dagger * Z^k
#             A_U = A[:, U_k, :]                 # (m, |U|, p)
#             W = t_pinv_apply(A_U, Z, rcond=rcond)   # (|U|, k, p)
#             update_z = t_product(A_U, W)          # (m, k, p)
#             Z = Z - t_product(A_U, W)          # (m, k, p)

#             # -----------------------------
#             # X-step: build J_k greedily using residual with Z^{k+1}
#             # eps_x^k = eta * max_i ||B - Z^{k+1} - A*X||_F^2 / ||A_i||_F^2
#             # -----------------------------
#             AX = t_product(A, X)               # (m, k, p)
#             R = B - Z - AX                     # (m, k, p)

#             scores_x = torch.zeros(m, device=A.device, dtype=A.dtype)
#             # for i in range(m):
#             #     R_i = R[i:i+1, :, :]           # (1, k, p)
#             #     scores_x[i] = (torch.sum(R_i ** 2)) / row_norms_sq[i]
#             for i in range(m):
#               # ||R_{i,:,:}||_F^2
#               R_i = R[i, :, :]
#               scores_x[i] = (torch.sum(R_i ** 2)) / row_norms_sq[i]

#             # eps_x = eta * torch.max(scores_x)
#             eps_x = eta * scores_x.max()
#             J_k = torch.where(scores_x >= eps_x)[0]

#             # Faithful block update:
#             # X^{k+1} = X^k + (A_J)^dagger * (B_J - Z_J - A_J*X^k)
#             A_J = A[J_k, :, :]                         # (|J|, n, p)
#             A_JX = t_product(A_J, X)                   # (|J|, k, p)
#             rhs = B[J_k, :, :] - Z[J_k, :, :] - A_JX   # (|J|, k, p)
#             dX = t_pinv_apply(A_J, rhs, rcond=rcond)   # (n, k, p)
#             X = X + dX

#             # monitoring
#             rse = rel_se(X, x_ls)
#             res_hist.append(rse.item())
#             x_hist.append(X.clone())

#             if rse < tol:
#                 break

#     runtime = time.time() - t0

#     X_np = X.detach().cpu().numpy()
#     return (X_np,iter_k + 1,np.array(res_hist),np.array([x.detach().cpu().numpy() for x in x_hist])), runtime