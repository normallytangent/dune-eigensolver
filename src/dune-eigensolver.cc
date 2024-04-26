// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

// always include the config file
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
// C++ includes
#include <math.h>
#include <iostream>
#include <vector>
#include <set>
#include <sstream>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

// dune-common includes
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/parametertreeparser.hh>
#include <dune/common/timer.hh>
#include <dune/common/exceptions.hh> // We use exceptions
// dune-istl includes
#include <dune/istl/preconditioners.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/pinfo.hh>
//#include <dune/istl/umfpack.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/istl/io.hh>
#include <dune/istl/eigenvalue/arpackpp.hh>
// eigensolver includes
#include "../dune/eigensolver/eigensolver.hh"

#include "../dune/eigensolver/arpack_geneo_wrapper.hh"

// global lock
double global_value;     // result
std::mutex global_mutex; // ensure exclusive access to sum

// barrier as a class
class Barrier
{
  int P;                                   // number of threads in barrier
  int count;                               // count number of threads that arrived at the barrier
  std::vector<int> flag;                   // flag indicating waiting thread
  std::mutex mx;                           // mutex for use with the cvs
  std::vector<std::condition_variable> cv; // for waiting
public:
  // set up barrier for given number of threads
  Barrier(int P_) : P(P_), count(0), flag(P_, 0), cv(P_)
  {
  }

  // get number of threads
  int nthreads() const
  {
    return P;
  }

  // wait at barrier
  void wait(int i)
  {
    // sequential case
    if (P == 1)
      return;

    std::unique_lock<std::mutex> ul{mx};
    count += 1; // one more
    if (count < P)
    {
      // wait on my cv until all have arrived
      flag[i] = 1; // indicate I am waiting
      cv[i].wait(ul, [i, this]
                 { return this->flag[i] == 0; }); // wait
    }
    else
    {
      // I am the last one, lets wake them up
      count = 0; // reset counter for next turn
      for (int j = 0; j < P; j++)
        if (flag[j] == 1)
        {
          flag[j] = 0;        // the event
          cv[j].notify_one(); // wake up
        }
    }
  }
};

/***************************************************
 *
 * Define some example matrices for the laplacian 
 * with different boundary conditions. 
 *
 ***************************************************/

 // Simulates the discretized meshes
Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> get_laplacian_dirichlet(int N)
{
  Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> A;
  int dummy = 0;
  setupLaplacian(A, N);
  return A;
}

Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> get_laplacian_neumann(int N)
{
  Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> A;
  setupLaplacian(A, N);
  for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
  {
    auto diag_iter = row_iter->begin();
    double value = 0.0;
    for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      if (row_iter.index() == col_iter.index())
        diag_iter = col_iter;
      else
        value += (*col_iter)[0][0];
    (*diag_iter)[0][0] = std::fabs(value);
  }
  return A;
}

Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> get_laplacian_B(int N, int overlap)
{
  Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> A;
  setupLaplacian(A, N);
  // @NOTE pu : partition of unity!
  std::vector<double> pu(A.N()); // pu is a vector of length N^2 of doubles.
  for (int k = 0; k < pu.size(); ++k)
  {
    int i = k / N;
    int j = k % N;
    if (i < overlap || i > N - 1 - overlap || j < overlap || j > N - 1 - overlap)
      pu[k] = 0.0;
    else
      pu[k] = 1.0;
  }
  for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      (*col_iter)[0][0] *= pu[row_iter.index()] * pu[col_iter.index()];
  return A;
}

Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> get_identity(int N)
{
  Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> A;
  setupLaplacian(A, N);
  for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      if (row_iter.index() == col_iter.index())
        (*col_iter)[0][0] = 1.0; // scaled the problem to check the type of tolerance(relative or absolute, if former, the error would have scaled by 10 as well!) (*col_iter)*10.0;
      else
        (*col_iter)[0][0] = 0.0;
  return A;
}

/***************************************************
 *
 * Do performance tests for Gram-Schmidt and Matmul
 *
 ***************************************************/
// @NOTE Focus on this test for comparison of the vanilla with AVX2
void mgs_performance_test(const Dune::ParameterTree &ptree, int rank, Barrier *pbarrier)
{
  std::cout<< "MGS STARTS HERE" << std::endl;
  // read parameters
  std::size_t n = ptree.get<std::size_t>("mgs.n");           // length of the vectors
  std::size_t m = ptree.get<std::size_t>("mgs.m");           // number of vectors
  std::size_t n_iter = ptree.get<std::size_t>("mgs.n_iter"); // number of iterations for test
  const std::size_t b = 8;                                   // block size

  if (rank == 0)
  {
    std::cout << "n=" << n << std::endl;
    std::cout << "m=" << m << std::endl;
    std::cout << "b=" << b << std::endl;
    std::cout << "n_iter=" << n_iter << std::endl;
  }

  // make a timer
  Dune::Timer timer;
  double time1;
  if (rank == 0)
    std::cout << "start test naive mgs version" << std::endl;
  {
    // allocate the matrix
    using MV = MultiVector<double, 1>;
    MV Q{n, m};

    // fill the Q matrix with random numbers
    auto const seed = 123;
    std::mt19937 urbg{seed};
    std::normal_distribution<double> generator{0.0, 1.0};
    for (std::size_t i = 0; i < Q.rows(); ++i)
      for (std::size_t j = 0; j < Q.cols(); j += 1)
        Q(i, j) = generator(urbg);

    pbarrier->wait(rank);
    timer.reset();
    for (int iter = 0; iter < n_iter; iter++)
      orthonormalize_naive(Q);
    pbarrier->wait(rank);
    time1 = timer.elapsed();
  }

  double time2;
  if (rank == 0)
    std::cout << "start test blocked mgs version" << std::endl;
  {
    // allocate the matrix
    using MV = MultiVector<double, b>;
    MV Q{n, m};

    // fill the Q matrix with random numbers
    auto const seed = 123;
    std::mt19937 urbg{seed};
    std::normal_distribution<double> generator{0.0, 1.0};
    for (std::size_t bj = 0; bj < Q.cols(); bj += MV::blocksize)
      for (std::size_t i = 0; i < Q.rows(); ++i)
        for (std::size_t j = 0; j < MV::blocksize; ++j)
          Q(i, bj + j) = generator(urbg);

    // run test
    pbarrier->wait(rank);
    timer.reset();
    for (int iter = 0; iter < n_iter; iter++)
      orthonormalize_blocked(Q);
    pbarrier->wait(rank);
    time2 = timer.elapsed();
  }

#ifdef VCINCLUDE
  double time3;
  if (rank == 0)
    std::cout << "start test avx2 mgs version" << std::endl;
  {
    // allocate the matrix
    using MV = MultiVector<double, b>;
    MV Q{n, m};

    // fill the Q matrix with random numbers
    auto const seed = 123;
    std::mt19937 urbg{seed};
    std::normal_distribution<double> generator{0.0, 1.0};
    for (std::size_t bj = 0; bj < Q.cols(); bj += MV::blocksize)
      for (std::size_t i = 0; i < Q.rows(); ++i)
        for (std::size_t j = 0; j < MV::blocksize; ++j)
          Q(i, bj + j) = generator(urbg);

    // run test
    pbarrier->wait(rank);
    timer.reset();
    for (int iter = 0; iter < n_iter; iter++)
      orthonormalize_avx2_b8(Q);
    pbarrier->wait(rank);
    time3 = timer.elapsed();
  }
#endif

#ifdef NEONINCLUDE
  double time3;
  if (rank == 0)
    std::cout << "start test VECTORIZED block mgs version (neon version)" << std::endl;
  {
    // allocate the matrix
    using MV = MultiVector<double, b>;
    MV Q{n, m};

    // fill the Q matrix with random numbers
    auto const seed = 123;
    std::mt19937 urbg{seed};
    std::normal_distribution<double> generator{0.0, 1.0};
    for (std::size_t bj = 0; bj < Q.cols(); bj += MV::blocksize)
      for (std::size_t i = 0; i < Q.rows(); ++i)
        for (std::size_t j = 0; j < MV::blocksize; ++j)
          Q(i, bj + j) = generator(urbg);

    // run test
    pbarrier->wait(rank);
    timer.reset();
    std::vector<double> dp;
    for (int iter = 0; iter < n_iter; iter++)
      orthonormalize_neon_b8_v2(Q);
    pbarrier->wait(rank);
    time3 = timer.elapsed();
  }
#endif

  if (rank == 0)
  {
    double flops = pbarrier->nthreads() * n_iter * flops_orthonormalize(n, m);
    double bytes = pbarrier->nthreads() * n_iter * bytes_orthonormalize_blocked(n, m, b, sizeof(double));
    double bytesn = pbarrier->nthreads() * n_iter * bytes_orthonormalize_naive(n, m, sizeof(double));
    std::cout << "P_n_m_i_iblocked_perfn_perfb_perfv "
              << pbarrier->nthreads() << " "
              << n << " "
              << m << " "
              << flops / bytesn << " "
              << flops / bytes << " "
              << flops / time1 * 1e-9 << " "
              << flops / time2 * 1e-9 << " "
#if defined(VCINCLUDE) || defined(NEOINCLUDE)
              << flops / time3 * 1e-9 << " "
#endif
              << std::endl;
  }

  std::cout<< "MGS ENDS HERE" << std::endl;
  return;
}

// @NOTE Implement this without islands matrix. Would be nice to see what
// percentage of the performance or time is spent in matrix vector multiplications
//int matvec_performance_test(const Dune::ParameterTree &ptree)
//{
//  try
//  {
//    auto istlA = get_islands_matrix(ptree);
//
//    std::size_t n = istlA.N();
//    std::size_t nnz = istlA.nonzeroes();
//    std::cout << "n=" << n << " nnz=" << nnz << std::endl;
//
//    std::size_t n_iter = ptree.get<std::size_t>("mv.n_iter"); // number of iterations for test
//    std::size_t n_vectors = ptree.get<std::size_t>("mv.m");
//    const std::size_t block_size = 8;
//
//    MultiVector<double, block_size> M1{n, n_vectors};
//    MultiVector<double, block_size> M2{n, n_vectors};
//
//    Dune::Timer timer;
//
//    auto const seed = 123;
//    std::mt19937 urbg{seed};
//    std::normal_distribution<double> generator{0.0, 1.0};
//    for (std::size_t bj = 0; bj < M1.cols(); bj += block_size)
//      for (std::size_t i = 0; i < M1.rows(); ++i)
//        for (std::size_t j = 0; j < block_size; ++j)
//          M1(i, bj + j) = generator(urbg);
//
//    // first version: naive matrix vector multiplication
//    std::cout << "mv naive version" << std::endl;
//    {
//      timer.reset();
//      for (int k = 0; k < n_iter; k++)
//      {
//        matmul_sparse_tallskinny_naive(M2, istlA, M1);
//        std::swap(M1, M2);
//      }
//      auto time1 = timer.elapsed();
//      std::cout << "elapsed time = " << time1 << std::endl;
//      double flops = 2.0 * n_iter * n_vectors * istlA.nonzeroes();
//      std::cout << "RESULT naive " << n << " " << nnz << " " << n_vectors << " " << flops * 1e-9 / time1 << std::endl;
//    }
//
//    // second version: multiple rhs matrix vector multiplication
//    std::cout << "mv block version" << std::endl;
//    {
//      for (std::size_t bj = 0; bj < M1.cols(); bj += block_size)
//        for (std::size_t i = 0; i < M1.rows(); ++i)
//          for (std::size_t j = 0; j < block_size; ++j)
//            M1(i, bj + j) = generator(urbg);
//      timer.reset();
//      for (int k = 0; k < n_iter; k++)
//      {
//        matmul_sparse_tallskinny_blocked(M2, istlA, M1);
//        std::swap(M1, M2);
//      }
//      auto time1 = timer.elapsed();
//      std::cout << "elapsed time = " << time1 << std::endl;
//      double flops = 2.0 * n_iter * n_vectors * istlA.nonzeroes();
//      std::cout << "RESULT blocked " << n << " " << nnz << " " << n_vectors << " " << flops * 1e-9 / time1 << std::endl;
//    }
//
//#ifdef VCINCLUDE
//    // third version: multiple rhs matrix vector multiplication
//    std::cout << "mv vectorized block version (Intel version)" << std::endl;
//    {
//      for (std::size_t bj = 0; bj < M1.cols(); bj += block_size)
//        for (std::size_t i = 0; i < M1.rows(); ++i)
//          for (std::size_t j = 0; j < block_size; ++j)
//            M1(i, bj + j) = generator(urbg);
//      timer.reset();
//      for (int k = 0; k < n_iter; k++)
//      {
//        matmul_sparse_tallskinny_avx2_b8(M2, istlA, M1);
//        std::swap(M1, M2);
//      }
//      auto time1 = timer.elapsed();
//      std::cout << "elapsed time = " << time1 << std::endl;
//      double flops = 2.0 * n_iter * n_vectors * istlA.nonzeroes();
//      std::cout << "RESULT avx2 " << n << " " << nnz << " " << n_vectors << " " << flops * 1e-9 / time1 << std::endl;
//    }
//#endif
//
//#ifdef NEONINCLUDE
//    // fourth version: multiple rhs matrix vector multiplication vectorized for arm
//    std::cout << "mv vectorized block version (arm version)" << std::endl;
//    {
//      for (std::size_t bj = 0; bj < M1.cols(); bj += block_size)
//        for (std::size_t i = 0; i < M1.rows(); ++i)
//          for (std::size_t j = 0; j < block_size; ++j)
//            M1(i, bj + j) = generator(urbg);
//      timer.reset();
//      for (int k = 0; k < n_iter; k++)
//      {
//        matmul_sparse_tallskinny_neon_b8(M2, istlA, M1);
//        std::swap(M1, M2);
//      }
//      auto time1 = timer.elapsed();
//      std::cout << "elapsed time = " << time1 << std::endl;
//      double flops = 2.0 * n_iter * n_vectors * istlA.nonzeroes();
//      std::cout << "RESULT neon " << n << " " << nnz << " " << n_vectors << " " << flops * 1e-9 / time1 << std::endl;
//    }
//#endif
//  }
//  catch (Dune::Exception &e)
//  {
//    std::cerr << "Dune reported error: " << e << std::endl;
//  }
//  catch (...)
//  {
//    std::cerr << "Unknown exception thrown!" << std::endl;
//  }
//  return 0;
//}

/***************************************************
 *
 * A first eigenvalue solver
 *
 ***************************************************/

// Returns a vector filled with all eigenvalues sorted in increasing size
std::vector<double> eigenvalues_laplace_dirichlet_2d(std::size_t N)
{
  std::vector<double> ev(N * N);
  double h = 1 / (N + 1.0);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      ev[j * N + i] = 4.0 * (std::sin(0.5 * h * (i + 1) * M_PI) * std::sin(0.5 * h * (i + 1) * M_PI) + std::sin(0.5 * h * (j + 1) * M_PI) * std::sin(0.5 * h * (j + 1) * M_PI));
  std::sort(ev.begin(), ev.end());
  return ev;
}

// Placeholder function. Needs figuring out the correct analytical solution to the neumann boundary value problem.
std::vector<double> eigenvalues_laplace_neumann_2d(int N)
{
  std::vector<double> ev(N * N);
  double h = 1 / (N + 1.0);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      ev[j * N + i] = 8.0 * (std::cosh(h * (i + 1) * M_PI) * std::cosh(h * (i + 1) * M_PI) + std::cosh(h * (j + 1) * M_PI) * std::cosh(h * (j + 1) * M_PI));

  std::sort(ev.begin(), ev.end());
  return ev;
}

/** \brief Calculate and print the residual of errors sampled in steps of ten.
* Rel = (computed_val - true_val) / true_val
*/
void RelativeResidual(std::vector<double> &eval,std::vector<double> &evalstop , std::vector<double> &true_eval)
{
  std::cout << std::endl;
  double reltolev = 0;
  double reltolevs = 0;
  for (int i = 0; i < eval.size(); i++){

    reltolev = ( eval[i] - true_eval[true_eval.size()-i-1] ) / true_eval[true_eval.size()-i-1];
    reltolevs = ( evalstop[i] - true_eval[true_eval.size()-i-1] ) / true_eval[true_eval.size()-i-1];

    if(i%10==0){
      std::cout << std::setw(16)
                << std::scientific
                << std::showpoint
                << std::setprecision(1)
                << abs(reltolev) << "   " << abs(reltolevs) << std::endl;
    }
  }
}

template <typename ISTLM>
void unshift_matrix(ISTLM &A, double shift)
{
  for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      if (row_iter.index() == col_iter.index())
        for (int i = 0; i < ISTLM::block_type::rows; i++)
          (*col_iter)[i][i] -= shift;
}

template <typename ISTLM>
void unshift_matrix(ISTLM &A, ISTLM &B, double shift, double regularization)
{
  A.axpy(-shift,B);
  for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      if (row_iter.index() == col_iter.index())
        for (int i = 0; i < ISTLM::block_type::rows; i++)
          (*col_iter)[i][i] -= regularization;
}

// this must be called sequentially
int largest_eigenvalues_convergence_test(const Dune::ParameterTree &ptree)
{
  std::cout << "Largest eigenvalues\n";
  // set up matrix
  int N = ptree.get<int>("evl.N");
  int overlap = ptree.get<int>("evl.overlap");
  auto A = get_laplacian_dirichlet(N);
  // auto A = get_laplacian_neumann(N);
  // auto B = get_laplacian_B(N, overlap);
  auto B = get_identity(N);
  using ISTLM = decltype(A);
  using block_type = typename ISTLM::block_type;
  // Dune::printmatrix(std::cout, B, "B", "");

  // obtain more parameters
  std::size_t br = block_type::rows;
  std::size_t bc = block_type::cols;
  std::size_t n = A.N() * br;
  int m = ptree.get<int>("evl.m");
  int maxiter = ptree.get<int>("evl.maxiter"); // number of iterations for test
  double shift = ptree.get<double>("evl.shift");
  double regularization = ptree.get<double>("evl.regularization");
  double tol = ptree.get<double>("evl.tol");
  int verbose = ptree.get<int>("evl.verbose");
  unsigned int seed = ptree.get<unsigned int>("evl.seed");
  std::string method = ptree.get<std::string>("evl.method");
  int stopperswitch = ptree.get<int> ("evl.stop");

  // first compute eigenvalues with arpack to great accuracy
  std::vector<double> eigenvalues_arpack(m, 0.0);
  using ISTLV = Dune::BlockVector<Dune::FieldVector<double, block_type::rows>>;
  ISTLV vec(n); //vec(A.N());
  vec = 0.0;
  std::vector<ISTLV> eigenvectors(m, vec);
  // arpack.computeGenSymShiftInvertMinMagnitude(B, 1e-14, eigenvectors, eigenvalues_arpack, -shift);
  ArpackMLGeneo::ArPackPlusPlus_Algorithms<ISTLM, ISTLV> arpack(A);
  arpack.computeStdSymMaxMagnitude(1e-14, eigenvectors, eigenvalues_arpack);

  // Now compute eigenvalues with given tolerance in arpack
  std::vector<double> eigenvalues_arpack2(m, 0.0);
  Dune::Timer timer_arpack;
  timer_arpack.reset();
  ArpackMLGeneo::ArPackPlusPlus_Algorithms<ISTLM, ISTLV> arpack_tol(A);
  arpack_tol.computeStdSymMaxMagnitude(tol, eigenvectors, eigenvalues_arpack2);
  auto time_arpack = timer_arpack.elapsed();
  auto arpackIterations = arpack_tol.getIterationCount();
  std::cout << ": arpack elapsed time " << time_arpack << std::endl;

  // Then compute the largest eigenvalue with ISTL's arpack wrapper
  Dune::ArPackPlusPlus_Algorithms<ISTLM, ISTLV> arp(A);
  double w = 0.0;
  arp.computeSymMaxMagnitude(tol,vec,w);
  std::cout << " " << w << std::endl;

  // Next compute eigenvalues with given tolerance in eigensolver
  std::vector<double> eval(m, 0.0);
  std::vector<std::vector<double>> evec(m);
  for (auto &v : evec)
    v.resize(n);
  double time_eigensolver;
  int esIterations;
  if (method == "std")
  {
    Dune::Timer timer_eigensolver;
    timer_eigensolver.reset();
    esIterations = StandardLargest(A, shift, tol, maxiter, m, eval, evec, verbose, seed);
    time_eigensolver = timer_eigensolver.elapsed();
  }

  // Also compute eigenvalues with given tolerance and new stopping criterion in eigensolver
  std::vector<double> evalstop(m, 0.0);
  std::vector<std::vector<double>> evecstop(m);
  for (auto &v : evecstop)
    v.resize(n);
  if (shift != 0.0)
    unshift_matrix(A, shift);

  double time_eigensolver_new_stopper;
  int essIterations;
  if (method == "std")
  {
    Dune::Timer timer_eigensolver_new_stopper;
    timer_eigensolver_new_stopper.reset();
    essIterations = StandardLargestWithNewStopper(A, shift, tol, maxiter, m, evalstop, evecstop, verbose, seed, stopperswitch);
    time_eigensolver_new_stopper = timer_eigensolver_new_stopper.elapsed();
  }
   // Finally compute eigenvalues for the 2d laplacian with dirichlet b.c.s. analytically
   std::vector<double> eigenvalues_analytical(N, 0.0);
   eigenvalues_analytical = eigenvalues_laplace_dirichlet_2d(N);

  // Printer
  std::cout << "eval_num" << std::setw(7) << " " << "EIGENSOLVER"
                                           << " " << "STOPCRITERION"
                                           << " " << " ANALYTICAL"
                                           << " " << " ARPACK-SMALL-TOL"
                                           << " " << " ARPACK-TOL"
                                           << " " << " ES-AN ERROR"
                                           << " " << " ES-AR ERROR"
                                           << " " << " ESS-AN ERROR"
                                           << " " << " ESS-AR ERROR"
                                           << std::endl;

   for (int i = 0; i < eval.size(); i++)
     std::cout << "eval[" << std::setw(3) << i << "]="
               << std::setw(16)
               << std::scientific
               << std::showpoint
               << std::setprecision(6)
               << eval[i]
               << "  "
               << evalstop[i]
               << "  "
               << eigenvalues_analytical[eigenvalues_analytical.size()-i-1]
               << "  "
               << eigenvalues_arpack[eigenvalues_arpack.size()-i-1]
               << "  "
               << eigenvalues_arpack2[eigenvalues_arpack2.size()-i-1]
               << "  "
               << std::abs(eval[i] - eigenvalues_analytical[eigenvalues_analytical.size()-i-1])
               << "  "
               << std::abs(eval[i] - eigenvalues_arpack[eigenvalues_arpack.size()-i-1])
               << "  "
               << std::abs(evalstop[i] - eigenvalues_analytical[eigenvalues_analytical.size()-i-1])
               << "  "
               << std::abs(evalstop[i] - eigenvalues_arpack[eigenvalues_arpack.size()-i-1])
               << std::endl;

  std::cout << "Relative residual of the stopping criterion with the arpack small tolerance\n";
  std::cout << "Eigensolver" << std::setw(20) << "StoppingCriterion";
  RelativeResidual(eval, evalstop, eigenvalues_arpack);

  std::cout << "Relative residual of the stopping criterion with the arpack\n";
  std::cout << "Eigensolver" << std::setw(20) << "StoppingCriterion";
  RelativeResidual(eval, evalstop, eigenvalues_arpack2);

  std::cout << "Relative residual of the stopping criterion with the analytical solution\n";
  std::cout << "Eigensolver" << std::setw(20) << "StoppingCriterion";
  RelativeResidual(eval, evalstop, eigenvalues_analytical);

  std::cout << "Relative residual of the arpack solutions with the analytical solution\n";
  std::cout << "Arpack small tol" << std::setw(16) << "Arpack";
  std::sort(eigenvalues_arpack.begin(),eigenvalues_arpack.end(),[](auto a, auto b)
                                                                 {
                                                                   return a > b;
                                                                 });
  std::sort(eigenvalues_arpack2.begin(),eigenvalues_arpack2.end(),[](auto a, auto b)
                                                                   {
                                                                     return a > b;
                                                                   });
  RelativeResidual(eigenvalues_arpack, eigenvalues_arpack2, eigenvalues_analytical);

  double maxerror = 0.0;
  for (int i = 0; i < eval.size(); i++)
    maxerror = std::max(maxerror, std::abs(eval[i] - eigenvalues_arpack[m-i-1]));

  double maxerror2 = 0.0;
  for (int i = 0; i < m; i++)
    maxerror2 = std::max(maxerror2, std::abs(eigenvalues_arpack2[m-i-1] - eigenvalues_arpack[m-i-1]));

   double maxerror3 = 0.0;
   for (int i = 0; i < m; ++i)
     maxerror3 = std::max(maxerror3, std::abs(eval[i]-eigenvalues_analytical[eigenvalues_analytical.size()-i-1]));

   double maxerror4 = 0.0;
   for (int i = 0; i < m; ++i)
     maxerror4 = std::max(maxerror4, std::abs(evalstop[i]-eigenvalues_analytical[eigenvalues_analytical.size()-i-1]));

  double maxerror5 = 0.0;
  for (int i = 0; i < eval.size(); i++)
    maxerror5 = std::max(maxerror, std::abs(evalstop[i] - eigenvalues_arpack[m-i-1]));


  std::cout << ": eigensolver elapsed time " << time_eigensolver << std::endl;
  std::cout << ": eigensolver with stopping criterion elapsed time " << time_eigensolver_new_stopper << std::endl;

  std::cout << "N" << "     " << "M"
                   << "     " << "TOL"
                   << "        " << "ESARERROR"
                   << "     " << "ESSTARERROR"
                   << "     " << "ARPERROR"
                   << "     " << "ESANERROR"
                   << "     " << "ESSTANERROR"
                   << "     " << "TIMERATIO"
                   << "     " << "TIMEWSTOPRATIO"
                   << "     " << "ARPACKITER"
                   << "     " << "ES_ITER"
                   << "     " << "ESS_ITER"
                   << std::endl;
  std::cout << n << "   "
            << m << "   "
            << tol << "    "
            << std::scientific
            << std::showpoint
            << std::setprecision(6)
            << maxerror << "      "
            << maxerror5 << "     "
            << maxerror2 << "      "
            << maxerror3 << "      "
            << maxerror4 << "      "
            << time_eigensolver / time_arpack << "      "
            << time_eigensolver_new_stopper / time_arpack << "     "
            << arpackIterations << "     "
            << esIterations << "      "
            << essIterations << " \\\\"
            << std::endl;

  return 0;
}

/***************************************************
 *
 * main function
 *
 ***************************************************/

int main(int argc, char **argv)
{
  // Maybe initialize MPI
  std::cout << "Hello World! This is dune-eigensolver." << std::endl;
  // DO NOT INITIALIZE MPI, it starts some threads!
  //  Dune::MPIHelper &helper = Dune::MPIHelper::instance(argc, argv);
  //  if (Dune::MPIHelper::isFake)
  //    std::cout << "This is a sequential program." << std::endl;
  //  else
  //    std::cout << "I am rank " << helper.rank() << " of " << helper.size()
  ////              << " processes!" << std::endl;

  // Read parameters from ini file
  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("dune-eigensolver.ini", ptree);
  ptreeparser.readOptions(argc, argv, ptree);

  const int P = std::thread::hardware_concurrency();
  int numthreads = ptree.get<int>("parallel.numthreads");
  std::cout << "hardware number of threads is " << P << " number of threads used is " << numthreads << std::endl;

   Barrier barrier(numthreads);
   std::vector<std::thread> threads;

  //  for (int rank = 0; rank < numthreads - 1; ++rank)
  //    threads.push_back(std::thread{mgs_performance_test, ptree, rank + 1, &barrier});
  //  mgs_performance_test(ptree, 0, &barrier);
  //  for (int rank = 0; rank < numthreads - 1; ++rank)
  //    threads[rank].join();

  // for (int rank = 0; rank < numthreads - 1; ++rank)
  //   threads.push_back(std::thread{eigenvalues_test, ptree, rank + 1, &barrier});
  // eigenvalues_test(ptree, 0, &barrier);
  // for (int rank = 0; rank < numthreads - 1; ++rank)
  //   threads[rank].join();

  std::cout << sizeof(int64_t) << " " << sizeof(long long) << std::endl;

  largest_eigenvalues_convergence_test(ptree);

  return 0;
}
