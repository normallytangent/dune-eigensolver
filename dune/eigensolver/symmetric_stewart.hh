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
  MultiVector<double, b> Q1{n, m}, Q2{n, m}, Q3{n, m}, Q4{n, m}, Se{m, m};

  //Initialize Raleigh coefficients
  std::vector<VEC> A_hat (m, VEC (m, 0.0));
  std::vector<VEC> B_hat (m, VEC (m, 0.0));

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
  std::cout << "reached inside while outer loop ";
    // Initialize with random numbers
    std::mt19937 urbg{seed};
    std::normal_distribution<double> generator{0.0, 1.0};
    for (std::size_t bj = (oiter > 0 ? ithelper : 0); bj < Q1.cols(); bj += b)
      for (std::size_t i = 0; i < Q1.rows(); ++i)
        for (std::size_t j = 0; j < b; ++j)
          Q1(i, bj + j) = generator(urbg);

    B_orthonormalize_blocked(B, Q1);

    Eigen::MatrixXd J(m, m), K(m, m), S(m, m);
    Eigen::VectorXd D(m);
    int iter = 0;
    double initial_partial_off = 0.0;
    while (iter < maxiter) 
    {
      matmul_sparse_tallskinny_blocked(Q3, B, Q1);
      matmul_inverse_tallskinny_blocked(Q1, F, Q3);
      B_orthonormalize_blocked(B, Q1);

      matmul_sparse_tallskinny_blocked(Q2, A, Q1);
      dot_products_all_blocked(A_hat, Q1, Q2);

    for( std::size_t i = 0; i < Q1.cols(); ++i)
      for (std::size_t j = 0; j < Q2.cols(); ++j)
        J(i, j) = A_hat[i][j];

    // Generalized selfadjoint eigen problem
    // Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(m);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ges(m);
    ges.compute(J);
    if( ges.info() != Eigen::Success)
      abort();
    D = ges.eigenvalues();
    S = ges.eigenvectors();
    
    auto peigen = &S(0,0);
    auto pmul = &Se(0,0);
      for (std::size_t i = 0; i < m ; ++i)
        pmul[i] = peigen[i];
    // show(pmul,m*m);
    // std::cout << S << "\n\n";
    // show(pmul, m,m);

    // Q1 = Q1 * S;
    matmul_tallskinny_dense_naive(Q4, Q1, Se);

    double partial_off = 0.0, partial_diag = 0.0;
    for (std::size_t i = 0; i < (size_t)Q2.cols()*0.75; ++i)
      for (std::size_t j = 0; j < (size_t)Q1.cols()*0.75; ++j)
        (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += A_hat[i][j] * A_hat[i][j];

    if (verbose > 0)
    std::cout << "# iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
    << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

    if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
      break;

    std::swap(Q1, Q4);
    ++iter;
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