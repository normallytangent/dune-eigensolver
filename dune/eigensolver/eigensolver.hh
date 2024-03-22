#ifndef Udune_eigensolver_HH
#define Udune_eigensolver_HH

#include "multivector.hh"
#include "umfpacktools.hh"
#include "kernels_cpp.hh"

//#define VCINCLUDE 1
//#define NEONINCLUDE 1

#ifdef VCINCLUDE
#include "kernels_avx2.hh"
#endif

#ifdef NEONINCLUDE
#include "kernels_neon.hh"
#endif

//#include "arpack_geneo_wrapper.hh"

/**  Randomized Eigensolver Package
 */

/** \brief solve starndard eigenvalue problem to obtain largest eigenvalues
 */
// @NOTE use power iteration for the largest eigval and orthogonal iteration for the n-largest eigenvalues
// @NEXTSTEPS This is the generalization of the power method, called Orthogonal Iteration 
template <typename ISTLM, typename VEC>
int StandardLargestWithNewStopper(ISTLM &A, double shift, double tol, int maxiter,
                                 int nev, std::vector<double> &eval,
                                 std::vector<VEC> &evec, int verbose = 0,
                                 unsigned int seed = 123,
                                 int stopperswitch = 0)
 {
    // types
    using block_type = typename ISTLM::block_type;

   //set the compile-time known block sizes for convenience
   const int b = 8;
   const int br = block_type::rows;
   const int bc = block_type::cols;
   if (br != bc)
     throw std::invalid_argument("StandardLargestWithNewStopper: blocks of input matrix must be square");

   // set the other sizes
   const std::size_t n = A.N() * br;
   const std::size_t m = ( nev / b + std::min(nev % b, 1)) * b; // make m the smallest possible multiple of the blocksize

   // allocate the two sets of vectors to iterate upon
   // allocate two tall and skinny matrices to iterate upon
   // n = number of rows, m = number of cols; must be a multiple of the block size!
   // assigns a pointer to an array of length n * m
   MultiVector<double, b> Q1{n, m};
   MultiVector<double, b> Q2{n, m};

   // initialize input vector with random numbers
   std::mt19937 urbg{seed};
   std::normal_distribution<double> generator{0.0, 1.0};
   for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
     for (std::size_t i = 0; i < Q1.rows(); ++i)
       for (std::size_t j = 0; j < b; ++j)
         Q1(i, bj + j) = generator(urbg);

   // apply shift; !loop overwrites input matrix A!
   // A = A + shift * I
   if (shift != 0.0)
   {
     for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
       for ( auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
         if (row_iter.index() == col_iter.index())
           for (int i = 0; i < br; i++)
             (*col_iter)[i][i] += shift; 
   }

   // orthonormalize the columns before starting iterations
   orthonormalize_blocked(Q1);

   // storage for Rayleigh quotients
   std::vector<double> s1(m, 0.0);

    double initial_norm = 0.0;
    int iter =0;
   // do iterations
   for (std::size_t k = 1; k < maxiter; ++k)
   {
     // Q2 = A*Q1
     matmul_sparse_tallskinny_blocked(Q2, A, Q1);

     // compute Rayleigh quotients
     // diag(D) = Q1T * Q2 
     dot_products_diagonal_blocked(s1, Q2, Q1);

     double frobenieus_norm = 0.0;
     if (stopperswitch == 0)
        // ||Q1T * Q2 ||i,j (i!=j) - tol * ||Q1T * Q2||i,i;
        frobenieus_norm = stopping_criterion_offdiagonal(tol, s1, Q1, Q2);
     else if (stopperswitch == 1)
        // || Q1 * diag(D) - Q2 ||;
        frobenieus_norm = stopping_criterion(s1, Q1, Q2);
     if (k == 1)
       initial_norm = frobenieus_norm;

     if (verbose > 0)
       std::cout << k << ": "<< frobenieus_norm << std::endl;

     // orthonormalize again for the next iteration
     orthonormalize_blocked(Q2);

     // exchange Q1 and Q2 for the next iteration
     std::swap(Q1, Q2);

     iter = k;

     if (k > 1 && frobenieus_norm < tol*initial_norm)
       break;
   }

   for (auto &x : s1)
  {
     std::cout << "old: " << x << std::endl;
     x -= shift*2;
     std::cout << "new: " << x << std::endl;
 }
   // store output
   // assumes that output is allocated to the correct size
   for (int j = 0; j < nev; ++j)
     eval[j] = s1[j];
   for (int j = 0; j < nev; ++j)
     for (int i = 0; i < n; ++i)
       evec[j][i] = Q1(i, j);
    return iter;
}

/** \brief solve starndard eigenvalue problem to obtain largest eigenvalues
 */
// @NOTE use power iteration for the largest eigval and orthogonal iteration for the n-largest eigenvalues
// @NEXTSTEPS This is the generalization of the power method, called Orthogonal Iteration 
template <typename ISTLM, typename VEC>
int StandardLargest(ISTLM &A, double shift, double tol, int maxiter, int nev, std::vector<double> &eval, std::vector<VEC> &evec, int verbose = 0, unsigned int seed = 123)
 {
    // types
    using block_type = typename ISTLM::block_type;

   //set the compile-time known block sizes for convenience
   const int b = 8;
   const int br = block_type::rows;
   const int bc = block_type::cols;
   if (br != bc)
     throw std::invalid_argument("StandardLargest: blocks of input matrix must be square");

   // set the other sizes
   const std::size_t n = A.N() * br;
   const std::size_t m = ( nev / b + std::min(nev % b, 1)) * b; // make m the smallest possible multiple of the blocksize

   // allocate the two sets of vectors to iterate upon
   // allocate two tall and skinny matrices to iterate upon
   // n = number of rows, m = number of cols; must be a multiple of the block size!
   // assigns a pointer to an array of length n * m
   MultiVector<double, b> Q1{n, m};
   MultiVector<double, b> Q2{n, m};

   // initialize input vector with random numbers
   std::mt19937 urbg{seed};
   std::normal_distribution<double> generator{0.0, 1.0};
   for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
     for (std::size_t i = 0; i < Q1.rows(); ++i)
       for (std::size_t j = 0; j < b; ++j)
         Q1(i, bj + j) = generator(urbg);

   // apply shift; !loop overwrites input matrix A!
   // A = A + shift * I
   if (shift != 0.0)
   {
     for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
       for ( auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
         if (row_iter.index() == col_iter.index())
           for (int i = 0; i < br; i++)
             (*col_iter)[i][i] += shift; 
   }

   // orthonormalize the columns before starting iterations
   orthonormalize_blocked(Q1);

   // storage for Rayleigh quotients
   std::vector<double> s1(m, 0.0), s2(m, 0.0);

   // do iterations
   int iter =0;
   for (std::size_t k = 1; k < maxiter; ++k)
   {
     // Q2 = A*Q1
     matmul_sparse_tallskinny_blocked(Q2, A, Q1);

     // compute Rayleigh quotients
     // diag(D) = Q1T * Q2 
     dot_products_diagonal_blocked(s1, Q2, Q1);
     for (auto &x : s1)
       x -= shift;

     // orthonormalize again
     orthonormalize_blocked(Q2);

     double distance = 0.0;
     for (int i = 0; i < s1.size(); i++)
       distance = std::max(distance, std::abs(s1[i] - s2[i]));
     if (verbose > 0 && k > 1)
       std::cout << "Iter=" << k << " " << distance << std::endl;
     std::swap(s1, s2);

     // exchange Q1 and Q2 for the next iteration
     std::swap(Q1, Q2);

    iter = k;
     // @NOTE Stopping criterion can be improved
     // We should think of a way to include the
     // check for convergence to the true solution.
     if (k > 1 && distance < tol)
       break;
   }

   // store output
   // assumes that output is allocated to the correct size
   for (int j = 0; j < nev; ++j)
     eval[j] = s2[j];
   for (int j = 0; j < nev; ++j)
     for (int i = 0; i < n; ++i)
       evec[j][i] = Q1(i, j); 
  return iter;
}

#endif // Udune_eigensolver_HH
