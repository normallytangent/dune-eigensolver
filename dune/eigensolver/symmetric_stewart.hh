#ifndef Udune_eigensolver_symmetric_stewart_HH
#define Udune_eigensolver_symmetric_stewart_HH

#include "eigensolver.hh"

template <typename ISTLM, typename VEC>
void GeneralizedSymmetricStewartAdaptive(const ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double tol, double threshold, int maxiter, int &nev,
                        VEC &eval, std::vector<VEC> &evec, 
                        int verbose = 0, unsigned int seed = 123)
{
  ISTLM A(inA);

  using block_type = typename ISTLM::block_type;
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br != bc)
    throw std::invalid_argument("GeneralizedSymmetricStewartAdaptive: blocks of input matrix must be square!");

  // Iteration vectors and matrices for eigenproblem in the subspace
  const std::size_t n = A.N() * br;

  std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible multiple of the blocksize
  MultiVector<double, b> Q1{n, m}, Q2{n, m}, Q3{n, m}, Q4{n, m}, Q5{n, m};

  //Initialize Raleigh coefficients
  std::vector<VEC> A_hat (Q2.cols(), VEC (Q1.cols(), 0.0));
  std::vector<VEC> B_hat (Q2.cols(), VEC (Q1.cols(), 0.0));

  // Apply shift and regularization
  if (shift != 0.0)
    A.axpy(shift, B);

  if (reg != 0.0)
  {
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        if (row_iter.index() == col_iter.index())
          for (int i = 0; i < br; i++)
            (*col_iter)[i][i] += reg;
  }

  // Compute factorization of matrix
  UMFPackFactorizedMatrix<ISTLM> F(A, std::max(0, verbose - 1));

  int initial_nev = m;
  bool finished = false;
  int oiter=0, ithelper = 0;
  while(!finished)
  {
    // Initialize with random numbers
    std::mt19937 urbg{seed};
    std::normal_distribution<double> generator{0.0, 1.0};
    for (std::size_t bj = (oiter > 0 ? ithelper : 0); bj < Q1.cols(); bj += b)
      for (std::size_t i = 0; i < Q1.rows(); ++i)
        for (std::size_t j = 0; j < b; ++j)
          Q1(i, bj + j) = generator(urbg);

#if defined (VCINCLUDE)
    B_orthonormalize_avx2_b8(B, Q1);
#else
    B_orthonormalize_blocked(B, Q1);
#endif

    Eigen::MatrixXd J(m, m), K(m, m), S(m, m);
    Eigen::VectorXd D(m);
    int iter = 0;
    double initial_partial_off = 0.0;
    while ( iter < maxiter)
    {
#if defined (VCINCLUDE)
      matmul_sparse_tallskinny_avx2_b8(Q3, B, Q1);
      matmul_inverse_tallskinny_avx2_b8(Q1, F, Q3);
      B_orthonormalize_avx2_b8(B, Q1);

#else
      matmul_sparse_tallskinny_blocked(Q3, B, Q1);
      matmul_inverse_tallskinny_blocked(Q1, F, Q3);
      B_orthonormalize_blocked(B, Q1);
#endif

#if defined (VCINCLUDE)
      matmul_sparse_tallskinny_avx2_b8(Q2, A, Q1);
#else
      matmul_sparse_tallskinny_blocked(Q2, A, Q1);
#endif
      dot_products_all_blocked(A_hat, Q1, Q2);

      for (size_t i = 0; i < Q1.cols(); ++i)
        for (size_t j = 0; j < Q1.cols(); ++j)
          J(i,j) = A_hat[i][j];

      Dune::Timer timer_eigendecomposition;
      // - SVD attempt: can't think of a way to use it right now since the eigenvalues and the eigenvectors are
      //    ordered from the largest to smallest.
      // - Using NoQRPreconditioner since matrix J is Square; reduces compile time, no affect on runtime. Source: Eigen doc JacobiSVD
      // - Maybe it does reduce the size of the executable code, but the accuracy gained per iteration is roughly the same as with
      //    Eigendecomposition.
      // Eigen::BDCSVD<Eigen::MatrixXd, Eigen::NoQRPreconditioner | Eigen::ComputeThinU | Eigen::ComputeFullV> svd;
      // svd.compute(J);
      // D = svd.singularValues().asDiagonal();
      // S = svd.matrixU();
      // std::cout <<" Full U: \n" << svd.matrixU() << std::endl;
      // std::cout << "Full V: \n" << svd.matrixV() << std::endl;

      // Eigendecomposition for symmetric matrix
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
      es.compute(J);
      if (es.info() != Eigen::Success)
        abort();
      D = es.eigenvalues(); 
      S = es.eigenvectors();

      // std::cout << "Eigendecomp S: \n" << S << std::endl;
      // time_eigendecomposition = timer_eigendecomposition.elapsed();
      // eigentimer += time_eigendecomposition;
    
      Eigen::MatrixXd EQ1(Q1.rows(), Q1.cols());
      Eigen::MatrixXd EQ2(Q1.rows(), Q1.cols());
      for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
        for (std::size_t i = 0; i < Q1.rows(); ++i)
          for (std::size_t j = 0; j < b; ++j)
            EQ1(i, bj + j) = Q1(i, bj+j);

      // Dune::Timer timer_matmul_dense;
      // matmul_tallskinny_dense_blocked(Q5, Q1, S);
      EQ2 = EQ1 * S;
      // time_matmul_dense = timer_matmul_dense.elapsed();
    
      for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
        for (std::size_t i = 0; i < Q1.rows(); ++i)
          for (std::size_t j = 0; j < b; ++j)
            Q5(i, bj + j) = EQ2(i, bj+j);
    
      // Stopping criterion
      double partial_off = 0.0, partial_diag = 0.0;
      for (std::size_t i = 0; i < (size_t)Q2.cols()*0.75; ++i)
        for (std::size_t j = 0; j < (size_t)Q1.cols()*0.75; ++j)
          (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += A_hat[i][j] * A_hat[i][j];

      if (verbose > 0)
        std::cout << "# iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
        << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

      if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
        break;

      Q1 = Q5;
      iter += 1;
    }
    
    // Stopping criterion given the desired eigenvalue is reached.
    if (D(initial_nev - 1) - shift >= threshold || initial_nev >= Q1.rows())
    {
      finished = true;
      std::cout << "Outer threshold executed! m, D(m - 1), threshold, oiter: " << initial_nev << "   " << D(initial_nev - 1) << "   " << threshold << "  " << oiter << "\n";
      if (eval.size() != m)
        eval.resize(m);
      for (size_t i = 0; i < m; ++i)
        eval[i] = D(i) - shift;
      break;
    }
    // Increase nev and provide an initial guess
    ithelper = m;
    m = std::min((int)Q1.rows(), (int)(initial_nev * 2));
    Q1.resize(m);
    Q2.resize(m);
    Q3.resize(m);
    Q4.resize(m);
    Q5.resize(m);

    A_hat.resize(m);
    for (auto &v : A_hat)
      v.resize(m, 0);

    initial_nev = m;

    ++oiter;
  }

  nev = m;


  if (verbose > 0)
  show(eval);

  if (evec.size() != m) {
    evec.resize(m);
    for (auto &v: evec)
      v.resize(n);
  }
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q1(i,j);
}

#endif