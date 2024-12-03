#!/usr/bin/env python
# -*- coding: utf-8 -*-
# refactoring standard stewart to use sparse linear algebra 
import numpy as np
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LaplacianNd, splu, inv, eigsh
from scipy.linalg import lu, blas, orth, eigh 
import math
import sys

def get_laplacian_rhs(N, overlap, problem):
    x_min = 0  #Left endpoint of x interval
    x_max = 1  #Right endpoint of x interval
    y_min = 0  #Left endpoint of y interval
    y_max = 1  #Right endpoint of y interval

    #mesh sizes and grid points
    nx = N
    ny = N
    n = nx*ny
    dx = (x_max - x_min)/(n+1)                    
    dy = (y_max - y_min)/(n+1)
    x = np.linspace(x_min+dx, x_max-dx, n)  # Grid points in x-direction
    y = np.linspace(y_min+dy, y_max-dy, n)  # Grid points in y-direction
    xg, yg = np.meshgrid(x, y)

    # Create the diagonals
    ones = np.full(n, 1)
    zeros = np.full(n, 0)

    a=np.ones(nx+ny-1,dtype='float')
    b=np.array([0.0],dtype='float')
    c=np.concatenate((a,b),axis=0)
    lower_diagonal = np.tile(c,nx)
    upper_diagonal = np.tile(c,nx)

    # Create the offsets
    offsets = [0, -1, 1, 4, -4]

    # Create the sparse matrix
    # A = diags([4*ones, -lower_diagonal, -upper_diagonal, -ones, -ones], offsets, shape=(n, n), dtype=float)

    # Set up A correctly for dirichlet
    # main_diag = np.ones(n)*4.0
    # side_diag = np.ones(n-1)
    # side_diag[np.arange(1,n)%nx==0] = 0
    # up_down_diag = np.ones(n-3)
    # diagonals = [main_diag, -side_diag, -side_diag, -up_down_diag, -up_down_diag]
    # laplacian = diags(diagonals, [0,-1,1,nx,-nx], format="csr", dtype=np.int8)
    # print(laplacian.toarray(), '\n\n')

    A_helper = LaplacianNd(grid_shape=(nx,nx), boundary_conditions=problem,dtype=np.float64)
    A = np.dot(-1,A_helper.toarray())
    A = csr_matrix(A)
    # Construct the RHS
    # B = diags([ones,zeros,zeros,zeros,zeros], offsets, shape=(n, n), dtype=np.float64).toarray()
    if problem == 'dirichlet':
        B = np.eye(n,dtype=np.float64)
    elif problem == 'neumann':
        B_helper = LaplacianNd((nx,nx), boundary_conditions='dirichlet',dtype=np.float64)
        B = np.dot(-1, B_helper.toarray())
        pu = np.zeros(n,dtype=np.int8)
        for k in range(0, pu.size):
            i = int(k / N)
            j = int(k % N)
            if (i < overlap) or (i > (N - 1 - overlap)) or (j < overlap) or (j > N - 1 - overlap):
                pu[k] = 0.0
            else:
                pu[k] = 1.0
        # print(pu)
        for itr, val in np.ndenumerate(B):
            B[itr] = val * pu[itr[0]] * pu[itr[1]]
    B = csr_matrix(B)
    #Solve the linear system using a sparse matrix solver
    # u = solve(Asdir, Bsdir, assume_a='pos').reshape(n, n)
    #Plot the solution
    # fig = plt.figure(figsize=(8, 8))
    # ax = plt.axes(projection='3d')
    # surf = ax.plot_surface(xg, yg, u)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('u')
    return A, B

def stewart_orthogonalize_std(A,m,shift,tol,maxiter,verbose):
    # shift A
    N = np.sqrt(A.shape[1]).astype(int)
    
    for i in range(N**2):
        for j in range(N**2):
            if i == j:
                A[i,j] += shift 
    # factorize A and calculate inverse
    PL, U = lu(A, permute_l=True)
    L_inv = np.linalg.inv(PL)
    U_inv = np.linalg.inv(U)
    A_inv = np.matmul(U_inv,L_inv)
    
    # setup randomized Eveq
    generator = np.random.Generator(np.random.MT19937(seed=400))
    Q_rand = generator.normal(0,1,size=(N**2,m))

    # orthonormalize Eveq
    Q, R = np.linalg.qr(Q_rand)
    # Loop:
    #   1. X := A^-1 * X
    #   2. Orthonormalize X
    #   3. A_hat := X^T * A * X
    #   4. A_hat =: S * D * S^T
    #   5. => D = (XS)^T * A * (XS) => X := XS
    D = np.zeros(m)
    Q2 = np.zeros((N**2, m))
    i = 0
    while i <  maxiter:
        Q1 = np.matmul(A_inv, Q)
        Q2, R = np.linalg.qr(Q1)
        Q3 = np.matmul(A, Q2)
        A_hat = np.matmul(Q2.T, Q3)
        D, S = np.linalg.eig(A_hat)
        # Theor. eigenvalues are 1 +/- 1e-9
        if verbose >= 2:
            print(S,'\n')
        Q = np.matmul(Q2, S)
        i+=1
    for i in range(m):
        D[i] -= shift 
    Eval = D
    Eveq = Q2
    
    return Eval, Eveq

def orthonormalize(Q_rand,m):
    Q, R = np.linalg.qr(Q_rand)
    P = orth(Q_rand)
    print("Shape of Q, R, P: ", Q.shape, R.shape, P.shape)
    # An orthogonal matrix is defined as the matrix product Q^-1 = Q_T
    print("Q^-1 = Q_T: ", np.allclose(np.linalg.inv(Q), Q.T))
    print("P^-1 = P_T: ", np.allclose(np.linalg.inv(P), P.T))
    # Given orthogonal matrix Q and a vector x: || Q . x ||_2 = || x ||_2
    print("\nQ_T * Q = I", np.allclose(Q.T@Q, np.eye(m)))
    print(Q.T@Q)
    print("\n||Q * x||_2 = ||x||_2: ", LA.norm(P@np.ones(m)), LA.norm(Q@np.ones(m)), LA.norm(np.ones(m)))
    print("\nGram-Schmidt test for Q_rand: ", np.allclose(Q @ R, Q_rand))
    return Q

def B_norm(M, x):
    a = math.sqrt( x.T @ M @ x )
    return a

def B_orthonormalize(B, A, m, verbose):
    #Q = A @NOTE creates a reference! not a by-value copy
    Q = np.zeros(A.shape, dtype=float)
    R = np.zeros((m,m),dtype=float)

    for k in range(0, Q.shape[1]):
        R[k,k] = B_norm(B, A[:,k])
        Q[:,k] = A[:,k] / R[k,k]
        for j in range(k+1, Q.shape[1]):
            R[k,j] = Q[:,k].T @ B @ A[:,j]
            A[:,j] = A[:,j] - Q[:,k] * R[k,j]

    # for j in range(0, Q.shape[1]):
    #     R[j,j] = B_norm(B, A[:,j])
    #     Q[:,j] = A[:,j] / R[j,j]
    #     for k in range(j+1, Q.shape[1]):
    #         R[j,k] = Q[:,j].T @ A[:,k] # ADD THE APPROPRIATE SCALAR PRODUCT!
    #         A[:,k] = A[:,k] - R[j,k] * Q[:,j]

    # Test correctness of orthonormalization
    if (Q.shape[0] == Q.shape[1] and verbose >=2):
        print("\n||Q * x||_B = ||x||_B: ", B_norm(B, np.matmul(Q,np.ones(m))), B_norm(B, np.ones(m)))
        print("\nQ_T * B * Q = I", np.allclose(Q.T@(B@Q), np.eye(m)), '\n', Q.T@(B@Q),'\n')
        print("\nQ^-1 = Q_T * B_T: ", np.allclose(np.linalg.inv(Q), Q.T@B.T))
        for j in range(0,Q.shape[1]):
            print(j,': ', B_norm(B,Q[:,j]))
    return Q

# def csr_allclose(a, b, rtol=1e-5, atol = 1e-8):
#     c = np.abs(np.abs(a - b) - rtol * np.abs(b))
#     return c.max() <= atol

def stewart_orthogonalize_gen(A,B,m,shift,tol,maxiter,verbose):
    N = np.sqrt(A.shape[1]).astype(int)
    A = A + shift * B

    # factorize A and calculate inverse
    LU = splu(A.tocsc())
    L = LU.L
    U = LU.U
    A_inv = inv(U) * inv(L)
    Ainv = inv(A.tocsc())

    print("A^-1 = U^-1 . L^-1 ", np.allclose( Ainv.todense(), A_inv.todense()))
    A_inv = A_inv.tocsr()
    Ainv = Ainv.tocsr()
    # A_Q, A_R = np.linalg.qr(A)
    # print("Gram-Schmidt test for A: ", np.allclose(A, A_Q@A_R))
    # print("A^-1 = R^-1 . Q^T ", np.allclose(np.linalg.inv(A), np.linalg.inv(A_R)@A_Q.T) )
    
    # setup randomized Q_rand
    generator = np.random.Generator(np.random.MT19937(seed=800))
    Q_rand = generator.normal(0,1,size=(N**2,m))

    # Orthonormalize Q with respect to the scalar product
    Q_orth = B_orthonormalize(B, Q_rand, m, verbose)

    # Loop:
    #   4. A_hat * S =: B_hat * S * D
    #       => A_hat = B_hat * S * D * S^-1, in most cases S will be invertible
    #   5. => D = (XS)^T * A * (XS) => X := XS
    initial_partial_off = 0.0
    D = np.zeros(m)
    Q0 = np.zeros((N**2, m))
    Q1 = np.zeros((N**2, m))
    Q2 = np.zeros((N**2, m))
    A_hat = np.zeros((m,m))
    B_hat = np.zeros((m,m))
    i = 0
    while i <  maxiter:
        if np.iscomplexobj(Q_orth):
            break
        else:
            Q0 = B * Q_orth
        Q1 = Ainv * Q0
        Q2 = B_orthonormalize(B, Q1, m, verbose)

        Q3 = np.zeros((N**2, m))
        Q3 = A * Q2
        np.matmul(Q2.T, Q3, out=A_hat)

        Q4 = np.zeros((N**2, m))
        Q4 = B * Q2
        np.matmul(Q2.T, Q4, out=B_hat)

        print('\nJ: ', A_hat, '\nK: ', B_hat, '\n\n')

        #Stewart Acceleration
        np.linalg.cholesky(A_hat)
        np.linalg.cholesky(B_hat)
        D, S = eigh(A_hat) # Theor. eigenvalues are 1 +/- 1e-9
        if verbose >= 2:
            print(S,'\n')
        np.matmul(Q2, S, out=Q_orth)

        # Stopping criterion computation
        partial_off = 0.0
        partial_diag = 0.0
        for j in range(0, math.ceil(m*0.75)):
            for k in range(0,math.ceil(m*0.75)):
                if j == k:
                    partial_diag += A_hat[j,k] * A_hat[j,k]
                elif j != k and i == 0:
                    initial_partial_off += A_hat[j,k] * A_hat[j,k]
                elif j != k:
                    partial_off += A_hat[j,k] * A_hat[j,k]
        if verbose >= 1:
            print('iter: ',i,' norm_off: %.6e' % np.sqrt(partial_off),' norm_diag: %.6e' % np.sqrt(partial_diag),'initial_norm_off: %.6e' % np.sqrt(initial_partial_off))
        # Stopping criterion condition
        if i > 0 and np.sqrt(partial_off) < tol * np.sqrt(initial_partial_off):
            break
        i+=1

    Eval = np.zeros((m),dtype=float)
    for i in range(m):
        Eval[i] = D[i] - shift
    Eveq = Q2.copy()
    
    for i in range(0, m):
        print("%.6e" % A_hat[i,i], ": ", "%.1e" % B_hat[i,i])

    return Eval, Eveq

def arpack_backend(A, B, m, shift, tol, maxiter):
    print('\n Arpack backend')
    W, E = eigsh(A, m, B, sigma=shift, tol=tol, maxiter=maxiter)
    # is_match = np.zeros((m),dtype=bool)
    # for i in range(0,m):
    #     # y = (W[i]) * np.matmul(B, E[:,i])
    #     # z = np.matmul(A, E[:,i])
    #     y = ((1/W[i]) * E[:,i])
    #     z = np.matmul(np.linalg.inv(A), np.matmul(B, E[:,i])) 
    #     is_match[i] = np.allclose(y,z)
    # print(all(is_match))

    return W, E

def main():
    N = 80 # 46
    m = 4 # 10
    verbose = 1
    shift = 1e-3
    tol = 1e-2
    maxiter = 2
    overlap = 1
    np.set_printoptions(threshold=sys.maxsize,linewidth=2000,formatter={'float': lambda x: format(x, '6.6e')})

    #### Dirichlet ####
    # Asdir, Bsdir = get_laplacian_rhs(N, overlap, 'dirichlet')
    # if verbose >= 2: 
    #     print(Asdir, '\n', Bsdir)
    # # Test if the symmetric matrices are positive definite!
    # np.linalg.cholesky(Asdir)
    # np.linalg.cholesky(Bsdir)

    # Eval_dir, Eveq_dir = stewart_orthogonalize_std(Asdir,m,shift,tol,maxiter,verbose)
    # if verbose >= 2:
    #     print('\n\n', Eveq_dir)
    # print(np.allclose(Asdir @ Eveq_dir, Eveq_dir @ np.diag(Eval_dir)))

    #### Neumann ####
    # Asneu = Asdir
    # Bsneu = Bsdir
    Asneu, Bsneu = get_laplacian_rhs(N, overlap, 'neumann')
    if verbose >= 2:
        print (Asneu, '\n\n', Bsneu)
    # np.linalg.cholesky(Asneu)
    # np.linalg.cholesky(Bsneu)
    Eval_neu, Eveq_neu = stewart_orthogonalize_gen(Asneu, Bsneu, m, shift, tol, maxiter, verbose)
    if verbose >= 2:
        print(Eval_neu, '\n\n', Eveq_neu)

    # Testing correctness
    W, E = arpack_backend(Asneu.todense(), Bsneu.todense(), m, shift, tol, maxiter)
    A = Asneu + shift * Bsneu
    # is_match = np.zeros((m),dtype=bool)
    # for i in range(0,m):
        # y = (Eval_neu[i]+shift) * np.matmul(Bsneu, Eveq_neu[:,i])
        # z = np.matmul(A, Eveq_neu[:,i])
        # print('\n',y,'\n',z,'\n')
        # is_match[i] = np.allclose(y,z)
    # print(is_match)
    # print(all(is_match))
    print(np.allclose(A.todense() @ Eveq_neu - Bsneu.todense() @ Eveq_neu @ (np.diag(Eval_neu)+shift * np.eye(m)), np.zeros((N**2, m))))

    # Printer
    print("## gen: stw\t\tArpack\t\tES-AR-ERROR")
    for i in range(0, m):
        print("%02i" %i, " %.6e" % Eval_neu[i], "\t %.6e" % W[i], "\t %.6e" % math.fabs(Eval_neu[i] - W[i]))
if __name__ == '__main__':
    main()
