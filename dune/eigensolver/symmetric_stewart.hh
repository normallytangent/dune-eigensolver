#ifndef Udune_eigensolver_symmetric_stewart_HH
#define Udune_eigensolver_symmetric_stewart_HH

#include "eigensolver.hh"

template <typename ISTLM, typename VEC>
void GeneralizedSymmetricStewartAdaptive(const ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double accuracy, double tol, double threshold, int maxiter, int &nev,
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

  // Measure total time
  Dune::Timer timer;

  // Iteration vectors and matrices for eigenproblem in the subspace
  const std::size_t n = A.N() * br;

  std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible multiple of the blocksize
  MultiVector<double, b> Q1{n, m}, Q3{n, m};

  //Initialize Raleigh coefficients
  std::vector<VEC> A_hat (Q1.cols(), VEC (Q1.cols(), 0.0));

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
  Dune::Timer timer_factorization;
  UMFPackFactorizedMatrix<ISTLM> F(A, std::max(0, verbose - 1));
  auto time_factorization = timer_factorization.elapsed();

  double time_eigendecomposition, time_matmul_dense;
  double eigentimer = 0, matmultimer = 0;
  int initial_nev = m;
  bool finished = false;
  int iter = 0, oiter=0, ithelper = 0;
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
    iter = 0;
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
      matmul_sparse_tallskinny_avx2_b8(Q3, A, Q1);
#else
      matmul_sparse_tallskinny_blocked(Q3, A, Q1);
#endif
      dot_products_all_blocked(A_hat, Q1, Q3);

      for (size_t i = 0; i < Q1.cols(); ++i)
        for (size_t j = 0; j < Q1.cols(); ++j)
          J(i,j) = A_hat[i][j];

      Dune::Timer timer_eigendecomposition;
      // Eigendecomposition for symmetric matrix
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
      es.compute(J);
      if (es.info() != Eigen::Success)
        abort();
      D = es.eigenvalues(); 
      S = es.eigenvectors();

      time_eigendecomposition = timer_eigendecomposition.elapsed();
      eigentimer += time_eigendecomposition;
    
      Eigen::MatrixXd EQ1(Q1.rows(), Q1.cols());
      Eigen::MatrixXd EQ2(Q1.rows(), Q1.cols());
      for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
        for (std::size_t i = 0; i < Q1.rows(); ++i)
          for (std::size_t j = 0; j < b; ++j)
            EQ1(i, bj + j) = Q1(i, bj+j);

      Dune::Timer timer_matmul_dense;
      // matmul_tallskinny_dense_blocked(Q3, Q1, S);
      EQ2 = EQ1 * S;
      time_matmul_dense = timer_matmul_dense.elapsed();
      matmultimer += time_matmul_dense;
    
      for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
        for (std::size_t i = 0; i < Q1.rows(); ++i)
          for (std::size_t j = 0; j < b; ++j)
            Q3(i, bj + j) = EQ2(i, bj+j);
    
      // Stopping criterion
      double partial_off = 0.0, partial_diag = 0.0;
      for (std::size_t i = 0; i < (size_t)Q1.cols()*accuracy; ++i)
        for (std::size_t j = 0; j < (size_t)Q1.cols()*accuracy; ++j)
          (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += A_hat[i][j] * A_hat[i][j];

      if (verbose > 0)
        std::cout << "# iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
        << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

      if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
        break;

      Q1 = Q3;
      iter += 1;
    }
    
    // Stopping criterion given the desired eigenvalue is reached.
    if (D(initial_nev - 1) - shift >= threshold || initial_nev >= Q1.cols())
    {
      finished = true;

      if (eval.size() != m)
        eval.resize(m);
      for (size_t i = 0; i < m; ++i)
        eval[i] = D(i) - shift;

      if (verbose > 0)
        std::cout << "Outer threshold executed! m, D(m - 1), threshold, oiter: " << initial_nev << "   " << D(initial_nev - 1) << "   " << threshold << "  " << oiter << "\n";
      break;
    }
    // Increase nev and provide an initial guess
    ithelper = m;
    m = std::min((int)Q1.cols(), (int)(initial_nev + 8));
    Q1.resize(m);
    Q3.resize(m);

    A_hat.resize(m);
    for (auto &v : A_hat)
      v.resize(m, 0);

    initial_nev = m;

    ++oiter;
  }

  nev = m;


  if (verbose > 1)
    show(eval);

  if (evec.size() != m) {
    evec.resize(m);
    for (auto &v: evec)
      v.resize(n);
  }
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q1(i,j);

  auto time = timer.elapsed();
  if (verbose > -1)
    std::cout << "# GeneralizedSymmetricStewartAdaptive: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_eigendecomposition=" << eigentimer
              << " time_matmul_dense=" << matmultimer
              << " iterations=" << ++iter
              << " outer_iterations=" << ++oiter
              << std::endl;
}

template <typename ISTLM, typename VEC>
void GeneralizedSymmetricStewart(const ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double accuracy, double tol, int maxiter, int nev,
                        VEC &eval, std::vector<VEC> &evec, 
                        int verbose = 0, unsigned int seed = 123, 
                        int stopperswitch=0)
{
  ISTLM A(inA);

  using block_type = typename ISTLM::block_type;
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br != bc)
    throw std::invalid_argument("GeneralizedSymmetricStewart: blocks of input matrix must be square!");

  // Measure total time
  Dune::Timer timer;

  const std::size_t n = A.N() * br;
  const std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible multiple of the blocksize

  // Iteration vectors
  MultiVector<double, b> Q1{n, m};
  MultiVector<double, b> Q3{n, m};
  MultiVector<double, b> Q4{n, m};

  // Initialize with random numbers
  std::mt19937 urbg{seed};
  std::normal_distribution<double> generator{0.0, 1.0};
  for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
    for (std::size_t i = 0; i < Q1.rows(); ++i)
      for (std::size_t j = 0; j < b; ++j)
        Q1(i, bj + j) = generator(urbg);

  // Apply shift; overwrites input matrix A
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
  Dune::Timer timer_factorization;
  UMFPackFactorizedMatrix<ISTLM> F(A, std::max(0, verbose - 1));
  auto time_factorization = timer_factorization.elapsed();

  // Eigen helper matrices
  Eigen::MatrixXd D (Q1.cols(), Q1.cols());
  Eigen::MatrixXd S (Q1.cols(), Q1.cols());
  Eigen::MatrixXd J(Q1.cols(), Q1.cols());
  Eigen::MatrixXd K(Q1.cols(), Q1.cols());

  //Initialize Raleigh coefficients
  std::vector<VEC> A_hat (Q1.cols(), VEC (Q1.cols(), 0.0));
  std::vector<VEC> B_hat (Q1.cols(), VEC (Q1.cols(), 0.0));

#if defined (VCINCLUDE)
  B_orthonormalize_avx2_b8(B, Q1);
#else
  B_orthonormalize_blocked(B, Q1);
#endif

  double eigentimer = 0;
  double initial_partial_off = 0.0;
  double time_eigendecomposition;
  double time_matmul_dense;
  int iter = 0;
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
    matmul_sparse_tallskinny_avx2_b8(Q3, A, Q1);
    // matmul_sparse_tallskinny_avx2_b8(Q4, B, Q1);
#else
    matmul_sparse_tallskinny_blocked(Q3, A, Q1);
    // matmul_sparse_tallskinny_blocked(Q4, B, Q1);
#endif
    dot_products_all_blocked(A_hat, Q1, Q3);
    // dot_products_all_blocked(B_hat, Q1, Q4);

    for (size_t i = 0; i < Q1.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
        J(i,j) = A_hat[i][j];
    for (size_t i = 0; i < Q1.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
        K(i,j) = B_hat[i][j];

    Dune::Timer timer_eigendecomposition;
    // Eigendecomposition for generalized real symmetric problem
    // Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
    // es.compute(J,K);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
    es.compute(J);
    if (es.info() != Eigen::Success)
      abort();
    D = es.eigenvalues(); 
    S = es.eigenvectors();

    time_eigendecomposition = timer_eigendecomposition.elapsed();
    eigentimer += time_eigendecomposition;
    
    Eigen::MatrixXd EQ1(Q1.rows(), Q1.cols());
    Eigen::MatrixXd EQ2(Q1.rows(), Q1.cols());
    for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
      for (std::size_t i = 0; i < Q1.rows(); ++i)
        for (std::size_t j = 0; j < b; ++j)
          EQ1(i, bj + j) = Q1(i, bj+j);

    Dune::Timer timer_matmul_dense;
    // matmul_tallskinny_dense_blocked(Q3, Q1, S);
    EQ2 = EQ1 * S;
    time_matmul_dense = timer_matmul_dense.elapsed();
    
    for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
      for (std::size_t i = 0; i < Q1.rows(); ++i)
        for (std::size_t j = 0; j < b; ++j)
          Q3(i, bj + j) = EQ2(i, bj+j);
    
    // Stopping criterion
    double partial_off = 0.0, partial_diag = 0.0;
    for (std::size_t i = 0; i < (size_t)Q1.cols()*accuracy; ++i)
      for (std::size_t j = 0; j < (size_t)Q1.cols()*accuracy; ++j)
        (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += A_hat[i][j] * A_hat[i][j];

    if (verbose > 0)
      std::cout << "# iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
      << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

    if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
      break;

    Q1 = Q3;
    iter += 1;
  }

  for (size_t i = 0; i < nev; ++i)
    eval[i] = D(i) - shift;
  // No need to sort in QR iteration
  // std::sort(eval.begin(),eval.end());

  for (int j = 0; j < nev; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q1(i, j);

  auto time = timer.elapsed();
  if (verbose > -1)
    std::cout << "# GeneralizedSymmetricStewart: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_eigendecomposition=" << eigentimer
              << " time_matmul_dense=" << time_matmul_dense
              << " iterations=" << ++iter
              << std::endl;
}

template <typename ISTLM, typename VEC>
void SymmetricStewart(ISTLM &inA, double shift, double accuracy, double tol, int maxiter,
                      int nev, VEC &eval, std::vector<VEC> &evec,
                      int verbose = 0, unsigned int seed = 123, int stopperswitch=0)
{
  ISTLM A(inA);

  using block_type = typename ISTLM::block_type;
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br != bc)
    throw std::invalid_argument("SymmetricStewart: blocks of input matrix must be square!");

  // Measure total time
  Dune::Timer timer;

  const std::size_t n = A.N() * br;
  const std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible multiple of the blocksize

  // Iteration vectors
  MultiVector<double, b> Q1{n, m};
  MultiVector<double, b> Q2{n, m};

  // Initialize with random numbers
  std::mt19937 urbg{seed};
  std::normal_distribution<double> generator{0.0, 1.0};
  for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
    for (std::size_t i = 0; i < Q1.rows(); ++i)
      for (std::size_t j = 0; j < b; ++j)
        Q1(i, bj + j) = generator(urbg);

  // Apply shift; overwrites input matrix A
  if (shift != 0.0)
  {
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        if (row_iter.index() == col_iter.index())
          for (int i = 0; i < br; i++)
            (*col_iter)[i][i] += shift;
  }

  // Compute factorization of matrix
  Dune::Timer timer_factorization;
  UMFPackFactorizedMatrix<ISTLM> F(A, std::max(0, verbose - 1));
  auto time_factorization = timer_factorization.elapsed();

  //Initialize Raleigh coefficients
  std::vector<VEC> A_hat (Q2.cols(), VEC (Q1.cols(), 0.0));
  std::vector<VEC> M_hat (Q2.cols(), VEC (Q1.cols(), 0.0));


  Eigen::MatrixXd D(Q2.cols(), Q1.cols());
  Eigen::MatrixXd S(Q2.cols(), Q1.cols());
  Eigen::MatrixXd B(Q2.cols(), Q1.cols());
  Eigen::MatrixXd C(Q2.cols(), Q1.cols());

#if defined (VCINCLUDE)
  orthonormalize_avx2_b8(Q1);
#else
  orthonormalize_blocked(Q1);
#endif

  double eigentimer = 0;
  double initial_partial_off = 0.0;
  double time_eigendecomposition;
  double time_matmul_dense;
  int iter = 0;
  while( iter < maxiter )
  {
#if defined (VCINCLUDE)
    matmul_inverse_tallskinny_avx2_b8(Q2, F, Q1);
    orthonormalize_avx2_b8(Q2);
#else
    matmul_inverse_tallskinny_blocked(Q2, F, Q1);
    orthonormalize_blocked(Q2);
#endif

#if defined (VCINCLUDE)
    matmul_sparse_tallskinny_avx2_b8(Q1, A, Q2);
#else
    matmul_sparse_tallskinny_blocked(Q1, A, Q2);
#endif
    dot_products_all_blocked(A_hat, Q2, Q1);

    for (size_t i = 0; i < Q2.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
          B(i,j) = A_hat[i][j];

    Dune::Timer timer_eigendecomposition;

    // Matrix decomposition of Q2T or B = S * D * S^T
    // Assumes a general standard eigenvalue problem
    Eigen::EigenSolver<Eigen::MatrixXd> es(B);
    if (es.info() != Eigen::Success)
      abort();
    D = es.pseudoEigenvalueMatrix();
    S = es.pseudoEigenvectors();

    time_eigendecomposition = timer_eigendecomposition.elapsed();
    eigentimer += time_eigendecomposition;

    Eigen::MatrixXd EQ1(Q1.rows(), Q1.cols());
    Eigen::MatrixXd EQ2(Q1.rows(), Q1.cols());
    for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
      for (std::size_t i = 0; i < Q1.rows(); ++i)
        for (std::size_t j = 0; j < b; ++j)
          EQ2(i, bj + j) = Q2(i, bj+j);

    Dune::Timer timer_matmul_dense;
    // matmul_tallskinny_dense_blocked(Q1, Q2, S); // Q1 = Q2 * Se;
    EQ1 = EQ2 * S;
    time_matmul_dense = timer_matmul_dense.elapsed();

    for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
      for (std::size_t i = 0; i < Q1.rows(); ++i)
        for (std::size_t j = 0; j < b; ++j)
          Q1(i, bj + j) = EQ1(i, bj+j);

    double partial_off = 0.0;
    double partial_diag = 0.0;
    // Stopping criterion
    for (std::size_t i = 0; i < (size_t)Q2.cols()*accuracy; ++i)
      for (std::size_t j = 0; j < (size_t)Q1.cols()*accuracy; ++j)
        (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += A_hat[i][j] * A_hat[i][j];

    if (verbose > 0)
      std::cout << "iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
      << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

    if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
      break;

    iter += 1;
  }

  for (size_t i = 0; i < nev; ++i)
    eval[i] = D(i,i) - shift;

  std::sort(eval.begin(),eval.end());

  if (verbose > 1)
    show(eval);

  for (int j = 0; j < nev; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q2(i, j);

  auto time = timer.elapsed();
  if (verbose > -1)
    std::cout << "# SymmetricStewart: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_eigendecomposition=" << eigentimer
              << " time_matmul_dense=" << time_matmul_dense
              << " iterations=" << ++iter
              << "\n";
}

#endif