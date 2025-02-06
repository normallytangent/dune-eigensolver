#ifndef Udune_eigensolver_HH
#define Udune_eigensolver_HH

#include <Eigen/Eigenvalues>
#include "multivector.hh"
#include "umfpacktools.hh"
#include "kernels_cpp.hh"
#define VCINCLUDE 1
//#define NEONINCLUDE 1

#ifdef VCINCLUDE
#include "kernels_avx2.hh"
#endif

#ifdef NEONINCLUDE
#include "kernels_neon.hh"
#endif

/**********************************
 *
 * Dune Eigensolver Package
 *
 **********************************/
/** \brief solve standard eigenvalue problem with shift invert to obtain
     smallest eigenvalues. Use the norm of the offdiagonal elements to stop.*/
template <typename ISTLM, typename VEC>
void StandardInverse(ISTLM &inA, double shift, double accuracy, double tol, int maxiter,
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
     else if (stopperswitch == 1)
     {
      // Stopping criterion
      for (std::size_t i = 0; i < (size_t)Q2.cols()*accuracy; ++i)
        for (std::size_t j = 0; j < (size_t)Q1.cols()*accuracy; ++j)
          (i == j ? partial_diag : partial_off) += Q2T[i][j] * Q2T[i][j];

      if (verbose > 1)
         std::cout << "iter: " << k << "; norm_off: "<< partial_off << "; norm_diag: " << partial_diag << "\n";

      if ( k > 0 && std::sqrt(partial_off) < tol * std::sqrt(partial_diag))
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

  if (stopperswitch == 0 || stopperswitch == 1 ){
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
  if (verbose > -1)
    std::cout << "# StandardInverse: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_dot_product_all=" << time_dot_product_all
              << " time_dot_product_diagonal=" << time_dot_product_diagonal
              << " iterations=" << ++k
              << std::endl;
}

/**  \brief solve standard eigenvalue problem with shift invert to obtain smallest eigenvalues
 *
 * Implementation assumes that A and B have the same sparsity pattern
 */
template <typename ISTLM, typename VEC>
void GeneralizedInverse(ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double accuracy, double tol, int maxiter, int nev,
                        VEC &eval, std::vector<VEC> &evec,
                        int verbose = 0, unsigned int seed = 123, int stopperswitch=0)
{
  ISTLM A(inA);

  using block_type = typename ISTLM::block_type;
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br != bc)
    throw std::invalid_argument("GeneralizedInverse: blocks of input matrix must be square!");

  // Measure total time
  Dune::Timer timer;

  const std::size_t n = A.N() * br;
  const std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible multiple of the blocksize

  // Iteration vectors
  MultiVector<double, b> Q1{n, m};
  MultiVector<double, b> Q2{n, m};
  MultiVector<double, b> Q3{n, m};

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

  //Initialize Raleigh coefficients
  std::vector<VEC> Q2T (Q2.cols(), VEC (Q1.cols(), 0.0));

#if defined (VCINCLUDE)
  B_orthonormalize_avx2_b8(B, Q1);
#else
  B_orthonormalize_blocked(B, Q1);
#endif

  double initial_partial_off = 0.0;
  int iter = 0;
  while ( iter < maxiter)
  {
    // Orthogonal iteration
#if defined (VCINCLUDE)
    matmul_sparse_tallskinny_avx2_b8(Q3, B, Q1);
    matmul_inverse_tallskinny_avx2_b8(Q1, F, Q3);
    B_orthonormalize_avx2_b8(B, Q1);
#else
    matmul_sparse_tallskinny_blocked(Q3, B, Q1);
    matmul_inverse_tallskinny_blocked(Q1, F, Q3);
    B_orthonormalize_blocked(B, Q1);
#endif

    // Rayleigh quotients
#if defined (VCINCLUDE)
    matmul_sparse_tallskinny_avx2_b8(Q2, A, Q1);
#else
    matmul_sparse_tallskinny_blocked(Q2, A, Q1);
#endif
    dot_products_all_blocked(Q2T, Q2, Q1);

    if (verbose > 1)
    {
      show(&(Q1(0,0)), Q1.rows(),Q1.cols());
      show(&(Q2(0,0)), Q2.rows(),Q2.cols());
      show(&(Q3(0,0)), Q3.rows(),Q3.cols());
    }

    // Stopping criterion
    double partial_off = 0.0;
    double partial_diag = 0.0;
    for (std::size_t i = 0; i < (size_t)Q2.cols()*accuracy; ++i)
      for (std::size_t j = 0; j < (size_t)Q1.cols()*accuracy; ++j)
        (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += Q2T[i][j] * Q2T[i][j];

    if (verbose > 0)
      std::cout << "# iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
      << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

    if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
      break;

  iter += 1;
  }

  for (size_t i = 0; i < nev; ++i)
    eval[i] = (Q2T[i][i]) - shift;

  if (verbose > 1)
    show(eval);

  std::sort(eval.begin(),eval.end());

  for (int j = 0; j < nev; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q1(i, j);

  auto time = timer.elapsed();
  if (verbose > -1)
    std::cout << "# GeneralizedInverse: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " iterations=" << ++iter
              << std::endl;
}

#endif // Udune_eigensolver_HH
