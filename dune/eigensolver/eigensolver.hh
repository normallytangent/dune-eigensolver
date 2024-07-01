#ifndef Udune_eigensolver_HH
#define Udune_eigensolver_HH

#include "multivector.hh"
#include "umfpacktools.hh"
#include "kernels_cpp.hh"
#include "../../../../external/eigen/build/include/eigen3/Eigen/Eigenvalues"
//#define VCINCLUDE 1
//#define NEONINCLUDE 1

#ifdef VCINCLUDE
#include "kernels_avx2.hh"
#endif

#ifdef NEONINCLUDE
#include "kernels_neon.hh"
#endif

//#include "arpack_geneo_wrapper.hh"

/**********************************
 *
 * Dune Eigensolver Package
 *
 **********************************/
/** \brief solve standard eigenvalue problem with shift invert to obtain
     smallest eigenvalues. Use the norm of the offdiagonal elements to stop.*/
template <typename ISTLM, typename VEC>
void StandardInverse(ISTLM &inA, double shift, double tol, int maxiter,
                                int nev, std::vector<double> &eval, 
                                std::vector<VEC> &evec, int verbose = 0,
                                unsigned int seed = 123, int stopperswitch=0)
{
  ISTLM A(inA);

  // types
  using block_type = typename ISTLM::block_type;

  // set the compile-time known block sizes for convenience
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br!=bc)
    throw std::invalid_argument("StandardInverseOffDiagonal: blocks of input matrix must be square");

  // Measure total time
  Dune::Timer timer;

  // set the other sizes
  const std::size_t n = A.N() * br;
  const std::size_t m = (nev / b + std::min(nev % b, 1)) * b; //= 32, make m the smallest possible  multiple of the blocksize

  // allocate the two sets of vectors to iterate upon
  MultiVector<double, b> Q1{n, m};
  MultiVector<double, b> Q2{n, m};

  // initialize with random numbers
  std::mt19937 urbg{seed};
  std::normal_distribution<double> generator{0.0, 1.0};
  for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
    for (std::size_t i = 0; i < Q1.rows(); ++i)
      for (std::size_t j = 0; j < b; ++j)
        Q1(i, bj + j) = generator(urbg);

  // apply shift; overwrites input matrix A
  if (shift != 0.0)
  {
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        if (row_iter.index() == col_iter.index())
          for (int i = 0; i < br; i++)
            (*col_iter)[i][i] += shift;
  }

  // compute factorization of matrix
  Dune::Timer timer_factorization;
  UMFPackFactorizedMatrix<ISTLM> F(A, std::max(0, verbose - 1));
  auto time_factorization = timer_factorization.elapsed();

  // orthonormalize the columns before starting iterations
  orthonormalize_blocked(Q1);

  // storage for Raleigh quotients
  std::vector<double> s1(m, 0.0), s2(m, 0.0);
  std::vector<std::vector<double>> Q2T (Q2.cols(), std::vector<double> (Q1.cols(),0.0));

  double initial_norm = 0.0;
  double time_dot_product_all, time_dot_product_diagonal;
  int k = 0;
  // do iterations
  for (k = 1; k < maxiter; ++k)
  {
    // Q2 = A^{-1}*Q1
    matmul_inverse_tallskinny_blocked(Q2, F, Q1); // apply inverse to all columns

    // orthonormalize again
    orthonormalize_blocked(Q2);

    // compute raleigh quotients
    // Q1 = A*Q2
    matmul_sparse_tallskinny_blocked(Q1, A, Q2);
    // diag(D) = Q2^T * Q1
    Dune::Timer timer_dot_product_all;
    dot_products_all_blocked(Q2T,Q2, Q1);
    time_dot_product_all = timer_dot_product_all.elapsed();
    Dune::Timer timer_dot_product_diagonal;
    dot_products_diagonal_blocked(s1, Q2, Q1);
    time_dot_product_diagonal = timer_dot_product_diagonal.elapsed();

     double frobenius_norm = 0.0;
     double partial_off = 0.0;
     double partial_diag = 0.0;
     if (stopperswitch == 0)
        // || Q2T * Q1 ||i,j (i!=j)  - tol * ||Q2T * Q1||i,i;
     {
       // frobenius_norm = stopping_criterion_offdiagonal(tol, s1, Q2, Q1);
      double norm = 0.0;
      for (std::size_t i = 0; i < Q2.cols(); ++i)
        for (std::size_t j = 0; j < Q1.cols(); ++j)
          if (i == j)
            partial_diag += Q2T[i][i] * Q2T[i][i];
          else
            partial_off += Q2T[i][j] * Q2T[i][j];

      if (verbose > 1)
         std::cout << k << ": "<< partial_off << "; " << partial_diag << std::endl;

      if (k > 1 && std::sqrt(partial_off) < tol * std::sqrt(partial_diag))
       break;
     }
     else if (stopperswitch == 2)
        // || Q1 * diag(D) - Q2 ||;
     {
       double distance = 0.0;
       for (int i = 0; i < s1.size(); i++)
         distance = std::max(distance, std::abs(s1[i] - s2[i]));
       if (verbose > 1)
         std::cout << k << ": " << distance << std::endl;
       std::swap(s1, s2);

       if (k == 1)
         initial_norm = frobenius_norm;

       if (k > 1 && distance < tol)
         break;
     }

    // exchange Q1 and Q2 for next iteration
    std::swap(Q1, Q2);
  }

  if (stopperswitch == 0 ){
    for (int j = 0; j < nev; ++j)
      eval[j] = s1[j] - shift;

    if (verbose > 1)
      show(eval);
  }
  else if (stopperswitch == 2){
    for (int j = 0; j < nev; ++j)
      eval[j] = s2[j] - shift;

    if (verbose > 1)
    show(eval);
  }

  // assumes that output is allocated to the correct size
  for (int j = 0; j < nev; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q1(i, j);

  auto time = timer.elapsed();
  if (verbose > 0)
    std::cout << "# StandardInverse: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_dot_product_all=" << time_dot_product_all
              << " time_dot_product_diagonal=" << time_dot_product_diagonal
              << " iterations=" << k
              << std::endl;
}

/**  \brief solve standard eigenvalue problem with shift invert to obtain smallest eigenvalues
 *
 * Implementation assumes that A and B have the same sparsity pattern
 */
template <typename ISTLM, typename VEC>
void GeneralizedInverse(ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double tol, int maxiter, int nev,
                        std::vector<double> &eval, std::vector<VEC> &evec,
                        int verbose = 0, unsigned int seed = 123, int stopperswitch=0)
{
  // copy matrix since we need to shift it
  ISTLM A(inA);

  // types
  using block_type = typename ISTLM::block_type;

  // set the compile-time known block sizes for convenience
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br != bc)
    throw std::invalid_argument("GeneralizedInverse: blocks of input matrix must be square");

  // measure time
  Dune::Timer timer;

  // set the other sizes
  const std::size_t n = A.N() * br;
  const std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible  multiple of the blocksize

  // allocate the two sets of vectors to iterate upon
  MultiVector<double, b> Q1{n, m};
  MultiVector<double, b> Q2{n, m};

  // initialize with random numbers
  std::mt19937 urbg{seed};
  std::normal_distribution<double> generator{0.0, 1.0};
  for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
    for (std::size_t i = 0; i < Q1.rows(); ++i)
      for (std::size_t j = 0; j < b; ++j)
        Q1(i, bj + j) = generator(urbg);

  // apply shift which means here A = A + shift*B; overwrites input matrix A
  // it is assumed that pattern(A) is included in pattern(B)
  if (shift != 0.0)
    A.axpy(shift, B);

  // add regularization; overwrites input matrix A
  if (reg != 0.0)
  {
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        if (row_iter.index() == col_iter.index())
          for (int i = 0; i < br; i++)
            (*col_iter)[i][i] += reg;
  }

  // compute factorization of matrix
  Dune::Timer timer_factorization;
  UMFPackFactorizedMatrix<ISTLM> F(A, std::max(0, verbose - 1));
  auto time_factorization = timer_factorization.elapsed();

  // B-orthonormalize and initialize Raleigh coefficients
  std::vector<double> ra1(m, 0.0), ra2(m, 0.0), sA(m, 0.0);
  std::vector<std::vector<double>> Q2T (Q2.cols(), std::vector<double> (Q1.cols(),0.0));
#if defined(NEONINCLUDE)
  B_orthonormalize_neon_b8(B, Q1);
  matmul_sparse_tallskinny_neon_b8(Q2, A, Q1);
  dot_products_diagonal_neon_b8(sA, Q2, Q1);
#elif defined(VCINCLUDE)
  B_orthonormalize_avx2_b8(B, Q1);
  matmul_sparse_tallskinny_avx2_b8(Q2, A, Q1);
  dot_products_diagonal_avx2_b8(sA, Q2, Q1);
#else
  double time_dot_product_all, time_dot_product_diagonal;
  B_orthonormalize_blocked(B, Q1);
  matmul_sparse_tallskinny_blocked(Q2, A, Q1);
  Dune::Timer timer_dot_product_all;
  dot_products_all_blocked(Q2T,Q2, Q1);
  time_dot_product_all = timer_dot_product_all.elapsed();
  Dune::Timer timer_dot_product_diagonal;
  dot_products_diagonal_blocked(sA, Q2, Q1);
  time_dot_product_diagonal = timer_dot_product_diagonal.elapsed();
#endif
  for (int i = 0; i < m; ++i)
    ra2[i] = sA[i] - shift;


  // do iterations
  double initial_norm = 0.0;
  int iter = 0;
  double relerror;
  while (iter < maxiter)
  {
    // Q2 = B*Q1; Q1 = A^{-1}*Q2
#if defined(NEONINCLUDE)
    matmul_sparse_tallskinny_neon_b8(Q2, B, Q1);
    matmul_inverse_tallskinny_neon_b8(Q1, F, Q2); // apply inverse to all columns
    B_orthonormalize_neon_b8(B, Q1);
#elif defined(VCINCLUDE)
    matmul_sparse_tallskinny_avx2_b8(Q2, B, Q1);
    matmul_inverse_tallskinny_avx2_b8(Q1, F, Q2); // apply inverse to all columns
    B_orthonormalize_avx2_b8(B, Q1);
#else
    matmul_sparse_tallskinny_blocked(Q2, B, Q1);
    matmul_inverse_tallskinny_blocked(Q1, F, Q2); // apply inverse to all columns
    B_orthonormalize_blocked(B, Q1);
#endif
    iter += 1;
    // compute raleigh quotients
#if defined(NEONINCLUDE)
    matmul_sparse_tallskinny_neon_b8(Q2, A, Q1);
    dot_products_diagonal_neon_b8(sA, Q2, Q1);
#elif defined(VCINCLUDE)
    matmul_sparse_tallskinny_avx2_b8(Q2, A, Q1);
    dot_products_diagonal_avx2_b8(sA, Q2, Q1);
#else
    matmul_sparse_tallskinny_blocked(Q2, A, Q1);
    dot_products_all_blocked(Q2T,Q2, Q1);
    dot_products_diagonal_blocked(sA, Q2, Q1);
#endif

    double frobenius_norm = 0.0;
    double partial_off = 0.0;
    double partial_diag = 0.0;
    if (stopperswitch == 0)
    {
      double norm = 0.0;
      for (std::size_t i = 0; i < Q2.cols(); ++i)
        for (std::size_t j = 0; j < Q1.cols(); ++j)
          if (i == j)
          {
            partial_diag += Q2T[i][i] *Q2T[i][i];
            //s1[i] = Q2T[i][i];
          }
          else
            partial_off += Q2T[i][j]*Q2T[i][j];

      if (verbose > 2)
        std::cout << iter << ": " << partial_off << "; " << partial_diag << std::endl;

      if (iter > 1 && std::sqrt(partial_off) < tol * std::sqrt(partial_diag))
        break;
    }
    else if (stopperswitch == 2)
    {
      for (int i = 0; i < m; ++i)
        ra1[i] = sA[i] - shift;
      if (verbose > 2)
        show(ra1);

      relerror = 0.0;
      for (int i = 0; i < m; ++i)
        relerror = std::max(relerror, std::abs(ra1[i] - ra2[i]));
      auto result = std::max_element(ra1.begin(), ra1.end());
      relerror /= *result;
      if (verbose > 2)
        std::cout << "iter=" << iter << " relerror=" << relerror << std::endl;
      std::swap(ra1, ra2);
      if (iter>10 & relerror < tol)
        break;
    }

  }

  for (int i = 0; i < m; ++i)
    ra1[i] = sA[i] - shift;
  if (verbose > 2)
    show(ra1);

  if (eval.size() != nev)
    eval.resize(nev);
  for (int j = 0; j < nev; ++j)
    eval[j] = ra1[j];

  if (evec.size() != nev)
    evec.resize(nev);
  for (int j = 0; j < nev; ++j)
  {
    if (evec[j].size() != n)
      evec[j].resize(n);
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q1(i, j);
  }

  auto time = timer.elapsed();
  if (verbose > 0)
    std::cout << "# GeneralizedInverse: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_dot_product_all=" << time_dot_product_all
              << " time_dot_product_diagonal=" << time_dot_product_diagonal
              << " iterations=" << iter
              << std::endl;
}

template <typename ISTLM, typename VEC>
void SymmetricStewart(ISTLM &inA, double shift,
                        double tol, int maxiter, int nev,
                        std::vector<double> &eval, std::vector<VEC> &evec,
                        std::vector<double> &analytical, int verbose = 0,
                        unsigned int seed = 123, int stopperswitch=0)
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
  MultiVector<double, b> Se{m, m};

  // Initialize with random numbers
  std::mt19937 urbg{seed};
  std::normal_distribution<double> generator{0.0, 1.0};
  for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
    for (std::size_t i = 0; i < Q1.rows(); ++i)
      for (std::size_t j = 0; j < b; ++j)
        Q2(i, bj + j) = generator(urbg);

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
  std::vector<double> s1(m, 0.0), s2(m, 0.0);
  std::vector<std::vector<double>> Q2T (Q2.cols(), std::vector<double> (Q1.cols(), 0.0));

  Eigen::MatrixXd S, D, B(Q2.cols(), Q1.cols());

  // Orthonormalize
  orthonormalize_blocked(Q2);

  double time_eigendecomposition;
  double time_matmul_dense;
  int iter = 0;
  for(iter = 1; iter < maxiter; ++iter)
  {
    matmul_inverse_tallskinny_blocked(Q1, F, Q2);
    orthonormalize_blocked(Q1);
    matmul_sparse_tallskinny_blocked(Q2, A, Q1); // Q2 = A * Q1
    dot_products_all_blocked(Q2T,Q1,Q2); // Q2T = Q1^T * Q2 =  Q1^T * A * Q1

    for (size_t i = 0; i < Q2.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
       B(i,j) = Q2T[i][j];

    if (verbose > 1)
    {
      std::cout << "BEFORE EIGEN" << std::endl;
      std::cout << B << std::endl << std::endl;
      show(&(Q2(0,0)), Q2.rows(),Q2.cols());
    }

    Dune::Timer timer_eigendecomposition;
    Eigen::EigenSolver<Eigen::MatrixXd> es(B); // Matrix decomposition of Q2T or B = S * D * S^T
    D = es.pseudoEigenvalueMatrix();
    S = es.pseudoEigenvectors();
    time_eigendecomposition = timer_eigendecomposition.elapsed();

    for (size_t i = 0; i < Q2.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
        Q2T[i][j] = B(i,j);

    for (size_t i = 0; i < Q2.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
        Se(i,j) = S(i,j);

    Dune::Timer timer_matmul_dense;
    matmul_tallskinny_dense_naive(Q2, Q2, Se); // Q2 = Q2 * Se;
    time_matmul_dense = timer_matmul_dense.elapsed();

    if (verbose > 1)
    {
      std::cout << "AFTER EIGEN" << std::endl;
      std::cout << B << std::endl << std::endl;
      show(&(Q2(0,0)), Q2.rows(),Q2.cols());
    }

  for (size_t i = 0; i < nev; ++i)
    D(i,i) = D(i,i) - shift;

    if (stopperswitch == 0)
    {
      double partial_off = 0.0;
      double partial_diag = 0.0;
      // Stopping criterion
      for (std::size_t i = 0; i < (size_t)Q2.cols()*0.3; ++i)
        for (std::size_t j = 0; j < (size_t)Q1.cols()*0.3; ++j)
          (i == j ? partial_diag : partial_off) += Q2T[i][j] * Q2T[i][j];

      if (verbose > 1)
         std::cout << iter << ": "<< partial_off << "; " << partial_diag << std::endl;

      if ( iter > 1 && std::sqrt(partial_off) < tol * std::sqrt(partial_diag))
        break;
    }
    else if (stopperswitch == 1)
    // analytical stopper switch based on the norm
    {
      double eq_norm = 0.0;
      for (size_t i = 0; i < nev - 10; ++i)
        eq_norm += ( D(i,i) - analytical[i] ) * (D(i,i) - analytical[i] );
      if( iter > 1 && std::sqrt(eq_norm) < tol)
        break;
    }

    std::swap(Q1, Q2);
  }

  for (size_t i = 0; i < nev; ++i)
    eval[i] = D(i,i); //- shift;

  if (verbose > 1)
    show(eval);

  std::sort(eval.begin(),eval.end());

  for (int j = 0; j < nev; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q2(i, j);

  auto time = timer.elapsed();
  if (verbose > 0)
    std::cout << "# SymmetricStewart: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_eigendecomposition=" << time_eigendecomposition
              << " time_matmul_dense=" << time_matmul_dense
              << " iterations=" << iter
              << std::endl;
}

template <typename ISTLM, typename VEC>
void GeneralizedSymmetricStewart(ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double tol, int maxiter, int nev,
                        std::vector<double> &eval, std::vector<VEC> &evec,
                        std::vector<double> &arpack, int verbose = 0,
                        unsigned int seed = 123, int stopperswitch=0)
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
  MultiVector<double, b> Q2{n, m};
  MultiVector<double, b> Se{m, m};

  // Initialize with random numbers
  std::mt19937 urbg{seed};
  std::normal_distribution<double> generator{0.0, 1.0};
  for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
    for (std::size_t i = 0; i < Q1.rows(); ++i)
      for (std::size_t j = 0; j < b; ++j)
        Q2(i, bj + j) = generator(urbg);

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

  //Initialize Raleigh coefficients
  std::vector<std::vector<double>> Q2T (Q2.cols(), std::vector<double> (Q1.cols(), 0.0));

  Eigen::MatrixXd S, D, E(Q2.cols(), Q1.cols());

  double time_eigendecomposition;
  double time_matmul_dense;
  int iter = 0;
  while ( iter < maxiter)
  {
#if defined(NEONINCLUDE)
    matmul_sparse_tallskinny_neon_b8(Q1, B, Q2);
    matmul_inverse_tallskinny_neon_b8(Q2, F, Q1);
    B_orthonormalize_neon_b8(B, Q2);
#elif defined(VCINCLUDE)
    matmul_sparse_tallskinny_avx2_b8(Q1, B, Q2);
    matmul_inverse_tallskinny_avx2_b8(Q2, F, Q1);
    B_orthonormalize_avx2_b8(B, Q2);
#else
    matmul_sparse_tallskinny_blocked(Q1, B, Q2);
    matmul_inverse_tallskinny_blocked(Q2, F, Q1);
    B_orthonormalize_blocked(B, Q2);
#endif

  iter += 1;

#if defined(NEONINCLUDE)
    matmul_sparse_tallskinny_neon_b8(Q1, A, Q2); // Q2 = A * Q1
    // dot_products_all_blocked(Q2T,Q2, Q1); // Q2T = Q1^T * Q2 =  Q1^T * A * Q1
#elif defined(VCINCLUDE)
    matmul_sparse_tallskinny_avx2_b8(Q1, A, Q2); // Q2 = A * Q1
    // dot_products_all_blocked(Q2T, Q2, Q1); // Q2T = Q1^T * Q2 =  Q1^T * A * Q1
#else
    matmul_sparse_tallskinny_blocked(Q1, A, Q2); // Q2 = A * Q1
    dot_products_all_blocked(Q2T, Q2, Q1); // Q2T = Q1^T * Q2 =  Q1^T * A * Q1
#endif

  for (size_t i = 0; i < Q2.cols(); ++i)
    for (size_t j = 0; j < Q1.cols(); ++j)
      E(i,j) = Q2T[i][j];

    if (verbose > 1)
    {
      std::cout << "BEFORE EIGEN" << std::endl;
      std::cout << E << std::endl << std::endl;
      show(&(Q1(0,0)), Q1.rows(),Q1.cols());
      show(&(Q2(0,0)), Q2.rows(),Q2.cols());
    }

    Dune::Timer timer_eigendecomposition;
    Eigen::EigenSolver<Eigen::MatrixXd> es(E); // Matrix decomposition of Q2T or B = S * D * S^T
    D = es.pseudoEigenvalueMatrix();
    S = es.pseudoEigenvectors();
    time_eigendecomposition = timer_eigendecomposition.elapsed();

    for (size_t i = 0; i < Q2.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
        Se(i,j) = S(i,j);

    Dune::Timer timer_matmul_dense;
    matmul_tallskinny_dense_naive(Q1, Q1, Se); // Q2 = Q2 * Se;
    time_matmul_dense = timer_matmul_dense.elapsed();

    if (verbose > 1)
    {
      std::cout << "AFTER EIGEN" << std::endl;
      std::cout << E << std::endl << std::endl;
      show(&(Q1(0,0)), Q1.rows(),Q1.cols());
      show(&(Q2(0,0)), Q2.rows(),Q2.cols());
    }

    for (size_t i = 0; i < nev; ++i)
      D(i,i) = D(i,i) - shift;

    if (stopperswitch == 0)
    {
      double partial_off = 0.0;
      double partial_diag = 0.0;
      // Stopping criterion
      for (std::size_t i = 0; i < (size_t)Q2.cols()*0.3; ++i)
        for (std::size_t j = 0; j < (size_t)Q1.cols()*0.3; ++j)
          (i == j ? partial_diag : partial_off) += Q2T[i][j] * Q2T[i][j];

      if (verbose > 1)
         std::cout << iter << ": "<< partial_off << "; " << partial_diag << std::endl;

      if ( iter > 1 && std::sqrt(partial_off) < tol * std::sqrt(partial_diag))
        break;
    }
    else if (stopperswitch == 1)
    // arpack stopper switch based on the norm
    {
      double eq_norm = 0.0;
      for (size_t i = 0; i < nev - 10; ++i)
        eq_norm += ( D(i,i) - arpack[i] ) * (D(i,i) - arpack[i] );
      std::cout << eq_norm << std::endl;
      if( iter > 1 && std::sqrt(eq_norm) < tol)
        break;
    }
    // std::swap(Q1, Q2);
  }

  for (size_t i = 0; i < nev; ++i)
    eval[i] = D(i,i);

  if (verbose > 1)
    show(eval);

  std::sort(eval.begin(),eval.end());

  for (int j = 0; j < nev; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q2(i, j);

  auto time = timer.elapsed();
  if (verbose > 0)
    std::cout << "# GeneralizedSymmetricStewart: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_eigendecomposition=" << time_eigendecomposition
              << " time_matmul_dense=" << time_matmul_dense
              << " iterations=" << iter
              << std::endl;
}

#endif // Udune_eigensolver_HH
