#ifndef Udune_eigensolver_kernels_cpp_HH
#define Udune_eigensolver_kernels_cpp_HH

#include "umfpacktools.hh"

//! simple dot product evaluation for comparison
template <typename MV>
std::vector<std::vector<double>> dot_products_diagonal(const MV &Q)
{
  std::vector<std::vector<double>> dp(Q.cols(), std::vector<double>(Q.cols(), 0.0));

  for (std::size_t i = 0; i < Q.cols(); i++)
    for (std::size_t j = 0; j < Q.cols(); j++)
    {
      double s = 0.0;
      for (std::size_t k = 0; k < Q.rows(); k++)
        s += Q(k, i) * Q(k, j);
      dp[i][j] = s;
    }
  return dp;
}

//! compute dot product of each column in Q1 with same column in Q2
template <typename MV>
void dot_products_diagonal_blocked(std::vector<double> &dp, const MV &Q1, const MV &Q2)
{
  if (dp.size() != Q1.cols())
    dp.resize(Q1.cols());
  if (Q1.rows() != Q2.rows())
    throw std::invalid_argument("dot_products_blocked: number of rows does not match");
  if (Q1.cols() != Q2.cols())
    throw std::invalid_argument("dot_products_blocked: number of columns does not match");
  std::size_t n = Q1.rows();
  std::size_t m = Q1.cols();
  std::size_t b = MV::blocksize;
  double s[b];

  for (std::size_t bj = 0; bj < m; bj += b)
  {
    auto q1 = &(Q1(0, bj));
    auto q2 = &(Q2(0, bj));
    for (std::size_t j = 0; j < b; ++j)
      s[j] = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < b; ++j)
        s[j] += q1[j] * q2[j];
      q1 += b;
      q2 += b;
    }
    for (std::size_t j = 0; j < b; ++j)
      dp[bj + j] = s[j];
  }
}

//! compute dot product of each column in Q1 with same column in Q2
template <typename MV>
void dot_products_all_blocked(std::vector<std::vector<double>> &dp, const MV &Q1, const MV &Q2)
{
  if (Q1.rows() != Q2.rows())
    throw std::invalid_argument("dot_products_blocked: number of rows does not match");
  if (Q1.cols() != Q2.cols())
    throw std::invalid_argument("dot_products_blocked: number of columns does not match");
  std::size_t n = Q1.rows();
  std::size_t m = Q1.cols();
  if (dp.size() != m)
    dp.resize(m);
  for (auto &v : dp)
    if (v.size() != m)
      v.resize(m);
  std::size_t b = MV::blocksize;
  double s[b][b];

  for (std::size_t bj1 = 0; bj1 < m; bj1 += b)
    for (std::size_t bj2 = 0; bj2 < m; bj2 += b)
    {
      auto q1 = &(Q1(0, bj1));
      auto q2 = &(Q2(0, bj2));
      for (std::size_t j1 = 0; j1 < b; ++j1)
        for (std::size_t j2 = 0; j2 < b; ++j2)
          s[j1][j2] = 0.0;

      for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t j1 = 0; j1 < b; ++j1)
          for (std::size_t j2 = 0; j2 < b; ++j2)
            s[j1][j2] += q1[j1] * q2[j2];
        q1 += b;
        q2 += b;
      }
      for (std::size_t j1 = 0; j1 < b; ++j1)
        for (std::size_t j2 = 0; j2 < b; ++j2)
          dp[bj1 + j1][bj2 + j2] = s[j1][j2];
    }
}

double flops_orthonormalize(int n, int m)
{
  double flops = 0.0;
  for (int k = m; k > 0; k--)
  {
    flops += 2 * n + n + (k - 1) * 4 * n;
  }
  return flops;
}

double bytes_orthonormalize_naive(int n, int m, int numbersize = 8)
{
  double numbers = 0.0;
  for (int k = m; k > 0; k--)
  {
    numbers += n + 2 * n + (k - 1) * (2 * n + 3 * n);
  }
  return numbers * numbersize;
}

/** @brief Orthogonalize a given MultiVector, naive version assuming block size 1
 *
 */
template <typename MV>
void orthonormalize_naive(MV &Q)
{
  if (MV::blocksize != 1)
    throw std::invalid_argument("orthonormalize_naive: blocksize must be one");
  using T = typename MV::value_type; // could be float or double
  auto n = Q.rows();
  auto m = Q.cols();
  auto q = &(Q(0, 0));
  for (std::size_t k = 0; k < m; ++k) // loop over all columns of Q
  {
    // std::cout << "scale k=" << k << std::endl;
    //  normalize q_k
    auto qk = &(q[k * n]); // data layout is one vector after the other
    double s = 0.0;        // compute norm of qk
    for (std::size_t i = 0; i < n; ++i)
      s += qk[i] * qk[i];
    s = 1.0 / std::sqrt(s);
    for (std::size_t i = 0; i < n; ++i)
      qk[i] *= s; // scale vector

    // make all remaining vectors orthogonal to qk
    for (std::size_t j = k + 1; j < m; ++j)
    {
      // std::cout << "  project j=" << j << std::endl;
      //  compute scalar product of column j>k with column k (which is already normalized)
      s = 0.0;
      auto qj = &(q[j * n]);
      for (std::size_t i = 0; i < n; ++i)
        s += qk[i] * qj[i];
      for (int i = 0; i < n; ++i)
        qj[i] -= s * qk[i];
    }
  }
}

double bytes_orthonormalize_blocked(int n, int m, int b, int numbersize = 8)
{
  double numbers = 0.0;
  // all blocks
  for (int bk = 0; bk < m; bk += b)
  {
    // diagonal block
    for (int k = b; k > 0; k--)
    {
      numbers += n * k + n * (1 + (k - 1) + 1);
    }
    // remaining blocks
    for (int bj = bk + b; bj < m; bj += b)
    {
      numbers += 5 * b * n;
    }
  }
  return numbers * numbersize;
}

/** @brief Orthogonalize a given MultiVector, block version for any block size relying on auto vectorization
 *
 */
template <typename MV>
void orthonormalize_blocked(MV &Q)
{
  using T = typename MV::value_type; // could be float or double
  std::size_t n = Q.rows();
  std::size_t m = Q.cols();
  auto q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;
  for (std::size_t bk = 0; bk < m; bk += b) // loop over all columns of Q in blocks
  {
    // std::cout << "bk=" << bk << std::endl;

    std::size_t nbk = bk * n; // start of block of b columns in memory

    // create memory for scalar products and initialize
    double s[b][b];
    for (std::size_t k = 0; k < b; ++k)
      for (std::size_t j = 0; j < b; ++j)
        s[k][j] = 0.0;

    // orthogonalization of diagonal block
    // std::cout << "  diagonal block" << std::endl;
    if (true)
    {
      for (std::size_t k = 0; k < b; ++k)
      {
        // compute scalar products of vectors in the block (upper triangle)
        // std::cout << "  scalar products" << std::endl;
        auto qbk = q + nbk;
        for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = k; j < b; ++j) // (m/b)*b*(b+1)/2   |  n*m*(b+1)/2
            s[k][j] += qbk[k] * qbk[j];
          qbk += b;
        }
        for (std::size_t j = k + 1; j < b; ++j)
          s[k][j] /= s[k][k];
        s[k][k] = 1.0 / std::sqrt(s[k][k]);

        // orthonormalize j>k with k and normalize
        qbk = &(q[nbk]);
        for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = k + 1; j < b; ++j) // n*m*(b+1)/2
            qbk[j] -= s[k][j] * qbk[k];
          qbk[k] *= s[k][k]; // normalization
          qbk += b;
        }
      }
    }
    else
    {
      // new block variant

      // compute scalar products
      auto vi = q + nbk;
      for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t k = 0; k < b; ++k)
          for (std::size_t j = k; j < b; ++j)
            s[k][j] += vi[k] * vi[j];
        vi += b;
      }
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = 0; j < k; ++j)
          s[k][j] = s[j][k];

      // do the LU factorization of s (might do Cholesky later)
      double LU[b][b];
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = 0; j < b; ++j)
          LU[k][j] = s[k][j];
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t i = k + 1; i < b; ++i)
        {
          LU[i][k] /= LU[k][k];
          for (std::size_t j = k + 1; j < b; ++j)
            LU[i][j] -= LU[i][k] * LU[k][j];
        }

      // pull out diagonal for later use: store the diagonal D^{-1/2}
      double D[b];
      for (std::size_t i = 0; i < b; ++i)
        D[i] = 1.0 / std::sqrt(LU[i][i]);

      // now we may prepare L in LU
      for (std::size_t i = 0; i < b; ++i)
      {
        LU[i][i] = 1.0;
        for (std::size_t j = i + 1; j < b; ++j)
          LU[i][j] = 0.0;
      }

      // compute inverse of L and store it in what will be U^T later
      double U[b][b];
      for (std::size_t i = 0; i < b; ++i) // initialize right hand side
        for (std::size_t j = 0; j < b; ++j)
          U[i][j] = (i == j) ? 1.0 : 0.0;
      for (std::size_t i = 1; i < b; ++i) // backsolve
        for (std::size_t j = 0; j < i; ++j)
          for (std::size_t k = 0; k < b; ++k)
            U[i][k] -= LU[i][j] * U[j][k];

      // transpose
      for (std::size_t i = 0; i < b; ++i)
        for (std::size_t j = 0; j < i; ++j)
          std::swap(U[i][j], U[j][i]);

      // finally scale by D^{-1/2} from the right
      for (std::size_t i = 0; i < b; ++i)
        for (std::size_t j = i; j < b; ++j)
          U[i][j] *= D[j];

      // now we can do the linear combinations
      vi = q + nbk;
      for (std::size_t i = 0; i < n; ++i)
      {
        for (int j = b - 1; j >= 0; --j)
        {
          double sum = 0.0;
          for (std::size_t k = 0; k <= j; ++k)
            sum += vi[k] * U[k][j]; // now we have consecutive access in V and U
          vi[j] = sum;
        }
        vi += b;
      }
    }

    // now do projection for all the remaining blocks with this block
    for (std::size_t bj = bk + b; bj < m; bj += b)
    {
      // std::cout << "    bj=" << bj << std::endl;

      std::size_t nbj = bj * n; // start of block of j columns in memory

      // clear scalar products
      for (std::size_t j = 0; j < b; ++j)
        for (std::size_t k = 0; k < b; ++k)
          s[k][j] = 0.0;

      // compute scalar products of all vectors in block bj with those in bk
      auto qbk = &(q[nbk]);
      auto qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t k = 0; k < b; ++k)
        {
#pragma omp simd
          for (std::size_t j = 0; j < b; ++j) //  (m/b)*(m/b-1)/2 * n*2*b
            s[k][j] += qbk[k] * qbj[j];
        }
        qbk += b;
        qbj += b;
      }

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t k = 0; k < b; ++k)
        {
#pragma omp simd
          for (std::size_t j = 0; j < b; ++j) //  (m/b)*(m/b-1)/2 * n*2*b
            qbj[j] -= s[k][j] * qbk[k];
        }
        qbk += b;
        qbj += b;
      }
    }
  }
}

/** @brief Orthogonalize a given MultiVector w.r.t. to scalar product given by sparse matrix B, block version for any block size relying on auto vectorization
 *
 */
template <typename ISTLM, typename MV>
double B_orthonormalize_blocked(const ISTLM &B, MV &Q)
{
  using block_type = typename ISTLM::block_type;
  const int br = block_type::rows;
  const int bc = block_type::cols;
  if (br != 1 || bc != 1)
    throw std::invalid_argument("B_orthonormalize_blocked: only implemented for FieldMatrix<..,1,1>");
  using T = typename MV::value_type; // could be float or double
  std::size_t n = Q.rows();
  std::size_t m = Q.cols();
  T *q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;
  double norm = 0.0; // compute maximum norm of strict upper tringle of R while we go

  // allocate space for one block of vectors to store B*vector
  T *p = new (std::align_val_t{64}) T[n * b]();

  // loop over all columns of Q in blocks
  for (std::size_t bk = 0; bk < m; bk += b)
  {
    std::size_t nbk = bk * n; // start of block of b columns in memory

    // now we need to compute B*(block of vectors) which we keep updated
    {
      T *pi = p;
      T *qbk = q + nbk;
      for (auto row_iter = B.begin(); row_iter != B.end(); ++row_iter)
      {
        for (std::size_t j = 0; j < b; ++j)
          pi[j] = 0.0;
        for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        {
          std::size_t J = col_iter.index() * b;
          for (std::size_t j = 0; j < b; ++j)
            pi[j] += (*col_iter) * qbk[J + j];
        }
        pi += b;
      }
    }

    // create memory for scalar products and initialize
    double s[b][b];
    for (std::size_t j = 0; j < b; ++j)
      for (std::size_t k = 0; k < b; ++k)
        s[k][j] = 0.0;

    if (false)
    {
      // // orthogonalization of diagonal block
      // for (std::size_t k = 0; k < b; ++k) // loop over all columns in this block
      // {
      //   // now we
      //   // compute scalar products of vectors in the block (upper triangle)
      //   T *qbk = q + nbk; // pointer to begin of block of vectors
      //   T *pi = p;
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     for (std::size_t j = k; j < b; ++j) // (m/b)*b*(b+1)/2   |  n*m*(b+1)/2
      //       s[k][j] += pi[k] * qbk[j];
      //     qbk += b;
      //     pi += b;
      //   }
      //   for (std::size_t j = k + 1; j < b; ++j)
      //     s[k][j] /= s[k][k];
      //   s[k][k] = 1.0 / std::sqrt(s[k][k]);
      //   for (std::size_t j = k + 1; j < b; ++j)
      //     norm = std::max(norm, s[k][j]);

      //   // orthonormalize j>k with k and normalize
      //   qbk = q + nbk;
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     for (std::size_t j = k + 1; j < b; ++j) // n*m*(b+1)/2
      //       qbk[j] -= s[k][j] * qbk[k];
      //     qbk[k] *= s[k][k]; // normalization
      //     qbk += b;
      //   }

      //   // update B*(block of vectors) in the same way
      //   pi = p;
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     for (std::size_t j = k + 1; j < b; ++j) // n*m*(b+1)/2
      //       pi[j] -= s[k][j] * pi[k];
      //     pi[k] *= s[k][k]; // normalization
      //     pi += b;
      //   }
      // }
    }
    else
    {
      // new block variant

      // compute B scalar products
      auto vi = q + nbk;
      T *pi = p;
      for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t k = 0; k < b; ++k)
          for (std::size_t j = k; j < b; ++j)
            s[k][j] += pi[k] * vi[j];
        vi += b;
        pi += b;
      }
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = 0; j < k; ++j)
          s[k][j] = s[j][k]; // extend lower triangle
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = k + 1; j < b; ++j)
          norm = std::max(norm, s[k][j]);

      // do the LU factorization of s (might do Cholesky later)
      double LU[b][b];
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = 0; j < b; ++j)
          LU[k][j] = s[k][j];
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t i = k + 1; i < b; ++i)
        {
          LU[i][k] /= LU[k][k];
          for (std::size_t j = k + 1; j < b; ++j)
            LU[i][j] -= LU[i][k] * LU[k][j];
        }

      // pull out diagonal for later use: store the diagonal D^{-1/2}
      double D[b];
      for (std::size_t i = 0; i < b; ++i)
        D[i] = 1.0 / std::sqrt(LU[i][i]);

      // now we may prepare L in LU
      for (std::size_t i = 0; i < b; ++i)
      {
        LU[i][i] = 1.0;
        for (std::size_t j = i + 1; j < b; ++j)
          LU[i][j] = 0.0;
      }

      // compute inverse of L and store it in what will be U^T later
      double U[b][b];
      for (std::size_t i = 0; i < b; ++i) // initialize right hand side
        for (std::size_t j = 0; j < b; ++j)
          U[i][j] = (i == j) ? 1.0 : 0.0;
      for (std::size_t i = 1; i < b; ++i) // backsolve
        for (std::size_t j = 0; j < i; ++j)
          for (std::size_t k = 0; k < b; ++k)
            U[i][k] -= LU[i][j] * U[j][k];

      // transpose
      for (std::size_t i = 0; i < b; ++i)
        for (std::size_t j = 0; j < i; ++j)
          std::swap(U[i][j], U[j][i]);

      // finally scale by D^{-1/2} from the right
      for (std::size_t i = 0; i < b; ++i)
        for (std::size_t j = i; j < b; ++j)
          U[i][j] *= D[j];

      // now update vectors
      vi = q + nbk;
      for (std::size_t i = 0; i < n; ++i)
      {
        for (int j = b - 1; j >= 0; --j)
        {
          double sum = 0.0;
          for (std::size_t k = 0; k <= j; ++k)
            sum += vi[k] * U[k][j]; // now we have consecutive access in V and U
          vi[j] = sum;
        }
        vi += b;
      }
      // and update the B vectors 
      pi = p;
      for (std::size_t i = 0; i < n; ++i)
      {
        for (int j = b - 1; j >= 0; --j)
        {
          double sum = 0.0;
          for (std::size_t k = 0; k <= j; ++k)
            sum += pi[k] * U[k][j]; // now we have consecutive access in V and U
          pi[j] = sum;
        }
        pi += b;
      }
    }

    // now do projection for all the remaining blocks with this block
    for (std::size_t bj = bk + b; bj < m; bj += b)
    {
      std::size_t nbj = bj * n; // start of block of j columns in memory

      // clear scalar products
      for (std::size_t j = 0; j < b; ++j)
        for (std::size_t k = 0; k < b; ++k)
          s[k][j] = 0.0;

      // compute scalar products of all vectors in block bj with those stored in p!
      auto pi = p;
      auto qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t k = 0; k < b; ++k)
        {
#pragma omp simd
          for (std::size_t j = 0; j < b; ++j) //  (m/b)*(m/b-1)/2 * n*2*b
            s[k][j] += pi[k] * qbj[j];
        }
        pi += b;
        qbj += b;
      }
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = 0; j < b; ++j)
          norm = std::max(norm, s[k][j]); // store r coefficient

      // do projections
      auto qbk = q + nbk;
      qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t k = 0; k < b; ++k)
        {
#pragma omp simd
          for (std::size_t j = 0; j < b; ++j) //  (m/b)*(m/b-1)/2 * n*2*b
            qbj[j] -= s[k][j] * qbk[k];
        }
        qbk += b;
        qbj += b;
      }
    }
  }

  // free memory
  delete[] p;

  return norm;
}

/** @brief multiply sparse matrix with tall skinny matrix stored in a multivector with block size 1
 *
 */
template <typename MV, typename ISTLM>
void matmul_sparse_tallskinny_naive(MV &Qout, const ISTLM &A, const MV &Qin)
{
  using block_type = typename ISTLM::block_type;
  const int br = block_type::rows;
  const int bc = block_type::cols;
  if (br != 1 || bc != 1)
    throw std::invalid_argument("matmul_sparse_tallskinny_naive: only implemented for FieldMatrix<..,1,1>");
  std::size_t n = Qin.rows();
  std::size_t m = Qin.cols();
  auto pin = &(Qin(0, 0));
  auto pout = &(Qout(0, 0));

  for (int j = 0; j < m; j++)
  {
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    {
      auto i = row_iter.index();
      pout[i] = 0.0;
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        pout[i] += (*col_iter) * pin[col_iter.index()];
    }
    pin += n;
    pout += n;
  }
}

/** @brief multiply sparse matrix with tall skinny matrix stored in a multivector
 *
 */
template <typename MV, typename ISTLM>
void matmul_sparse_tallskinny_blocked(MV &Qout, const ISTLM &A, const MV &Qin)
{
  using block_type = typename ISTLM::block_type;
  const int br = block_type::rows;
  const int bc = block_type::cols;
  if (br != 1 || bc != 1)
    throw std::invalid_argument("matmul_sparse_tallskinny_blocked: only implemented for FieldMatrix<..,1,1>");
  std::size_t n = Qin.rows();
  std::size_t m = Qin.cols();
  std::size_t block_size = MV::blocksize;
  auto pin = &(Qin(0, 0));
  auto pout = &(Qout(0, 0));

  for (std::size_t bj = 0; bj < m; bj += block_size)
  {
    std::size_t nbj = n * bj;
    std::size_t I = n * bj;
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    {
      for (std::size_t j = 0; j < block_size; ++j)
        pout[I + j] = 0.0;
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      {
        std::size_t J = nbj + col_iter.index() * block_size;
        for (std::size_t j = 0; j < block_size; ++j)
          pout[I + j] += (*col_iter) * pin[J + j];
      }
      I += block_size;
    }
  }
}

/** @brief Calculate the norm of the resulting eigenvectors in order to measure the convergence to solution.
 *
*/
template <typename MV>
double stopping_criterion(std::vector<double> &dp,const MV &Q1, const MV &Q2) {
  std::size_t b = MV::blocksize;

  double partial = 0.0;
  double norm = 0.0;
  for (std::size_t bj = 0; bj < Q1.cols(); bj += b)
    for (std::size_t i = 0; i < Q1.rows(); ++i)
      for (std::size_t j = 0; j < b; ++j)
        partial += (Q1(i, bj+j)*dp[bj+j] - Q2(i,bj+j))*(Q1(i, bj+j)*dp[bj+j] - Q2(i,bj+j));

  norm = std::sqrt(partial);
  return norm;
}

//! Apply inverse in factorized form; you may overwrite the input argument
template <typename MV, typename MAT>
void matmul_inverse_tallskinny_blocked(MV &Qout, UMFPackFactorizedMatrix<MAT> &F, MV &Qin)
{
  // std::cout << "enter matmul_inverse_tallskinny_blocked" << std::endl;
  if (Qout.rows() != Qin.rows() || Qout.cols() != Qin.cols())
    throw std::invalid_argument("matmul_inverse_tallskinny_blocked: Qout/Qin size mismatch");
  if (F.n != Qin.rows() || F.n != Qout.rows())
    throw std::invalid_argument("matmul_inverse_tallskinny_blocked: Factorization does not match size of Qout/Qin");

  const int n = Qin.rows(); // rows in Q as well as size of square factorized matrix
  const int m = Qin.cols(); // columns in Q
  const int block_size = MV::blocksize;
  auto pin = &(Qin(0, 0));
  auto pout = &(Qout(0, 0));

  // loop over all blocks of columns
  for (int bj = 0; bj < m; bj += block_size)
  {
    int nbj = n * bj;

    // combined row scaling and permutation
    // store result in Qout
    if (F.do_recip)
    {
      int K = nbj;                // first index of block
      for (int k = 0; k < n; ++k) // loop over rows
      {
        auto scaling = F.Rs[F.P[k]];
        int I = nbj + F.P[k] * block_size; // permuted row start
        for (int s = 0; s < block_size; ++s)
          pout[K + s] = scaling * pin[I + s];
        K += block_size;
      }
    }
    else
    {
      int K = nbj;                // first index of block
      for (int k = 0; k < n; ++k) // loop over rows
      {
        auto scaling = 1.0 / F.Rs[F.P[k]];
        int I = nbj + F.P[k] * block_size; // permuted row start
        for (int s = 0; s < block_size; ++s)
          pout[K + s] = scaling * pin[I + s];
        K += block_size;
      }
    }

    // back solve L (in compressed row storage)
    // rhs are in Qout
    // store result in Qin
    {
      int I = nbj; // first index of block
      double sum[block_size];
      for (int i = 0; i < n; i++) // loop over all rows
      {
        for (int s = 0; s < block_size; ++s)
          sum[s] = pout[I + s]; // load with right hand side
        for (auto k = F.Lp[i]; k < F.Lp[i + 1] - 1; k++)
        {
          int J = nbj + F.Lj[k] * block_size;
          auto lij = F.Lx[k];
          for (int s = 0; s < block_size; ++s)
            sum[s] -= lij * pin[J + s];
        }
        for (int s = 0; s < block_size; ++s)
          pin[I + s] = sum[s];
        I += block_size;
      }
    }

    // back solve U (in compressed column storage)
    // input in Qin, output in Qin (so that we can store the permutation in Qout in the end)
    {
      int J = nbj + (n - 1) * block_size; // last small row in this block
      double result[block_size];          // the new xj's
      for (int j = n - 1; j >= 0; j--)
      {
        double matelem = F.Ux[F.Up[j + 1] - 1];
        for (int s = 0; s < block_size; ++s)
          result[s] = pin[J + s] / matelem;
        for (auto k = F.Up[j]; k < F.Up[j + 1] - 1; k++)
        {
          int I = nbj + F.Ui[k] * block_size;
          matelem = F.Ux[k];
          for (int s = 0; s < block_size; ++s)
            pin[I + s] -= matelem * result[s];
        }
        // store result at permuted index
        int K = nbj + F.Q[j] * block_size;
        for (int s = 0; s < block_size; ++s)
          pout[K + s] = result[s];
        J -= block_size;
      }
    }
  }
}



#endif
