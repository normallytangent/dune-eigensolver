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
#include <dune/istl/umfpack.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/istl/io.hh>
// eigensolver includes
#include "../dune/eigensolver/eigensolver.hh"

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


/***************************************
 *
 * do performance tests for Gram-Schmidt
 *
 ***************************************/
// @NOTE Focus on this test for comparison of the vanilla with AVX2
void mgs_performance_test(const Dune::ParameterTree &ptree, int rank, Barrier *pbarrier)
{
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

  return;
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

   for (int rank = 0; rank < numthreads - 1; ++rank)
     threads.push_back(std::thread{mgs_performance_test, ptree, rank + 1, &barrier});
   mgs_performance_test(ptree, 0, &barrier);
   for (int rank = 0; rank < numthreads - 1; ++rank)
     threads[rank].join();

  return 0;
}
