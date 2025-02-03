#ifndef Udune_eigensolver_nonsymmetric_stewart_HH
#define Udune_eigensolver_nonsymmetric_stewart_HH

#include "eigensolver.hh"

void sort_schur(Eigen::MatrixXd & T, Eigen::MatrixXd & U) 
{
    int n = T.rows();
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i)
        indices[i] = i;

    // compare the magnitude of eigenvalues
    auto comp = [&T](int a, int b) {
        return std::abs(T(a, a)) < std::abs(T(b, b));
    };

    // sort indices based on the comparison
    std::sort(indices.begin(), indices.end(), comp);

    Eigen::MatrixXd T_sorted = T;
    Eigen::MatrixXd U_sorted = U;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            T_sorted(i, j) = T(indices[i], indices[j]);
            U_sorted(i, j) = U(i, indices[j]);
        }

    T = T_sorted;
    U = U_sorted;
}

template <typename ISTLM, typename VEC>
void GeneralizedNonsymmetricStewart(const ISTLM &inA, const ISTLM &B, double shift,
                        double reg, double tol, int maxiter, int nev,
                        std::vector<double> &eval, std::vector<VEC> &evec, 
                        int verbose = 0, unsigned int seed = 123, 
                        int stopperswitch=0)
{
  ISTLM A(inA);

  using block_type = typename ISTLM::block_type;
  const int b = 8;
  const int br = block_type::rows; // = 1
  const int bc = block_type::cols; // = 1
  if (br != bc)
    throw std::invalid_argument("GeneralizedNonSymmetricStewart: blocks of input matrix must be square!");

  // Measure total time
  Dune::Timer timer;

  const std::size_t n = A.N() * br;
  const std::size_t m = (nev / b + std::min(nev % b, 1)) * b; // = 32, make m the smallest possible multiple of the blocksize

  // Iteration vectors
  MultiVector<double, b> Q1{n, m};
  MultiVector<double, b> Q2{n, m};
  MultiVector<double, b> Q3{n, m};
  MultiVector<double, b> Q4{n, m};
  MultiVector<double, b> Q5{n, m};
  MultiVector<double, b> Q6{n, m};
  MultiVector<double, b> U1{m, m};

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

  Eigen::MatrixXd D (Q2.cols(), Q1.cols());
  Eigen::MatrixXd S (Q2.cols(), Q1.cols());
  Eigen::MatrixXd U (Q2.cols(), Q1.cols());
  Eigen::MatrixXd T (Q2.cols(), Q1.cols());
  //Initialize Raleigh coefficients
  std::vector<std::vector<double>> A_hat (Q2.cols(), std::vector<double> (Q1.cols(), 0.0));
  std::vector<std::vector<double>> B_hat (Q2.cols(), std::vector<double> (Q1.cols(), 0.0));

  B_orthonormalize_blocked(B, Q1);
//   matmul_sparse_tallskinny_blocked(Q2, B, Q1);
//   matmul_inverse_tallskinny_blocked(Q3, F, Q2);
  // std::vector<double> eigentimer()
  double eigentimer = 0;
  double initial_partial_off = 0.0;
  double time_eigendecomposition;
  double time_matmul_dense;
  int iter = 0, t = 6;
  double k = 0;
  while ( iter < maxiter)
  {     
    // while () {
        // while () { 
            // 1. "A*Q"
            matmul_sparse_tallskinny_blocked(Q2, B, Q1);
            matmul_inverse_tallskinny_blocked(Q3, F, Q2);
        // }
        // 2. Orthogonalize "A*Q"
        B_orthonormalize_blocked(B, Q3);
    // }
    // 3. Compute the SRR Approximation
    matmul_sparse_tallskinny_blocked(Q4, A, Q3);
    dot_products_all_blocked(A_hat, Q3, Q4);

    for (size_t i = 0; i < Q1.cols(); ++i)
      for (size_t j = 0; j < Q1.cols(); ++j)
        S(i,j) = A_hat[i][j];
    Dune::Timer timer_eigendecomposition;
    Eigen::RealSchur<Eigen::MatrixXd> schur(m);
    schur.compute(S);
    if (schur.info() != Eigen::Success)
      abort();
    T = schur.matrixT();
    U = schur.matrixU();
    sort_schur(T,U);

    // k <= t / std::log10(T.norm()*(T.inverse()).norm());
    // Q4.norm() <= Q1 T // AQ = QT 

    // (4.3)
    auto pQ3 = &Q3(0,0);
    for (size_t g = 0; g < Q3.cols()*Q3.rows();++g)
        pQ3[g] *= T(0,0);
        
    Q5 = Q4 - Q3; // Residual matrix
    // std::cout << Q5 << std::endl;

    time_eigendecomposition = timer_eigendecomposition.elapsed();
    eigentimer += time_eigendecomposition;
    
    // Stopping criterion
    // double partial_off = 0.0, partial_diag = 0.0;
    // for (std::size_t i = 0; i < (size_t)Q2.rows(); ++i)
    //   for (std::size_t j = 0; j < (size_t)Q1.cols()*0.75; ++j)
    //     (i == j ? partial_diag : (iter == 0 ? initial_partial_off : partial_off)) += Q4(i,j) * Q4(i,j);

    // if (verbose > 0)
    //   std::cout << "# iter: " << iter << " norm_off: "<< std::sqrt(partial_off) << " norm_diag: " << std::sqrt(partial_diag)
    //   << " initial_norm_off: " << std::sqrt(initial_partial_off) << "\n";

    // if ( iter > 0 && std::sqrt(partial_off) < tol * std::sqrt(initial_partial_off))
    //   break;
    std::cout << T << std::endl;
    std::cout << std::endl;
    std::cout << U << std::endl;
    std::cout << std::endl;
    show(&U1(0,0), U1.rows(), U1.cols());
    std::cout << std::endl;
    for (std::size_t bj = 0; bj < U1.rows(); bj += b)
      for (std::size_t i = 0; i < U1.cols(); ++i)
        for (std::size_t j = 0; j < b; ++j)
          U1(i, bj + j) = U(bj + j, i);

    matmul_tallskinny_dense_naive(Q6, Q3, U1);
    iter += 1;
    Q1 = Q6;
  }
  auto time = timer.elapsed();

  for (size_t i = 0; i < nev; ++i)
    eval[i] = T(i,i) - shift;

  for (int j = 0; j < nev; ++j)
    for (int i = 0; i < n; ++i)
      evec[j][i] = Q1(i, j);

  if (verbose > 0)
    std::cout << "# GeneralizedNonSymmetricStewart: "
              << " time_total=" << time
              << " time_factorization=" << time_factorization
              << " time_eigendecomposition=" << eigentimer //time_eigendecomposition
              << " time_matmul_dense=" << time_matmul_dense
              << " iterations=" << iter
              << std::endl;
}
#endif