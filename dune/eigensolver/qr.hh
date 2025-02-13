#ifndef Udune_eigensolver_qr_HH
#define Udune_eigensolver_qr_HH

#include "eigensolver.hh"

template <typename ISTLM, typename VEC>
void GeneralizedInverseAdaptive(const ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double accuracy, double tol, double threshold, int maxiter, int &nev,
                        VEC &eval, std::vector<VEC> &evec, 
                        int verbose = 0, unsigned int seed = 123)
{
  ISTLM A(inA);

  Dune::Timer timer;

  using block_type = typename ISTLM::block_type;
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br != bc)
    throw std::invalid_argument("GeneralizedInverseAdaptive: blocks of input matrix must be square!");

  // Iteration vectors and matrices for eigenproblem in the subspace
  const std::size_t n = A.N() * br;

  std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible multiple of the blocksize

  // The initial search space can only be the lesser of the 
  // blocksize and the size requested by the user.
  std::size_t initial_nev = std::min((int) b, (int) m);

  MultiVector<double, b> Q1{n, initial_nev}, Q3{n, initial_nev};

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

  bool finished = false;
  int iter =0, oiter=0, ithelper = 0;
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

    iter = 0;
    double initial_partial_off = 0.0;
    while (iter < maxiter) 
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

    double partial_off = 0.0, partial_diag = 0.0;
    for (std::size_t i = 0; i < (size_t)Q1.cols()*accuracy; ++i)
      for (std::size_t j = 0; j < (size_t)Q1.cols()*accuracy; ++j)
        (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += A_hat[i][j] * A_hat[i][j];

    if (verbose > 0)
      std::cout << "# iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
    << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

    if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
      break;

    ++iter;
    }
    
    // Stopping criterion given the desired eigenvalue is reached.
    if (A_hat[initial_nev - 1][initial_nev - 1] - shift >= threshold || initial_nev >= evec.size())
    {
      finished = true;
      if(verbose > 0)
        std::cout << "Outer threshold executed! m, D(m - 1), threshold, oiter: " << initial_nev << "   " << A_hat[initial_nev -1][initial_nev - 1] << "   " << threshold << "  " << oiter << "\n";
      break;
    }
    // Increase nev and provide an initial guess
    ithelper = initial_nev;
    initial_nev = std::min((int)evec.size(), (int)(initial_nev + b));
    Q1.resize(initial_nev);
    Q3.resize(initial_nev);

    A_hat.resize(initial_nev);
    for (auto &v : A_hat)
      v.resize(initial_nev, 0);

    ++oiter;
  }

  m = initial_nev;

  if (eval.size() != m)
    eval.resize(m);
  for (size_t i = 0; i < m; ++i)
    eval[i] = A_hat[i][i] - shift;

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
    std::cout << "# GeneralizedInverseAdaptive: "
              << std::scientific
              << std::showpoint
              << std::setprecision(6)
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " iterations=" << ++iter
              << " outer_iterations=" << ++oiter
              << std::endl;
}

#endif