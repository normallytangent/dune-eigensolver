#ifndef Udune_eigensolver_kernels_avx2_HH
#define Udune_eigensolver_kernels_avx2_HH

#include "umfpacktools.hh"
#include "../../../../external/vcl/vectorclass.h"

//! compute dot product of each column in Q1 with same column in Q2; vectorized for neon
template <typename MV>
void dot_products_diagonal_avx2_b8(std::vector<double> &dp, const MV &Q1, const MV &Q2)
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
  Vec4d S0, S1; // registers to store scalar products
  Vec4d X0, X1;
  Vec4d Y0, Y1;

  for (std::size_t bj = 0; bj < m; bj += b)
  {
    // pointers to start of blocks
    auto q1 = &(Q1(0, bj));
    auto q2 = &(Q2(0, bj));

    // clear summation variable
    S0 = Vec4d(0.0);
    S1 = Vec4d(0.0);

    for (std::size_t i = 0; i < n; ++i)
    {
      X0.load(q1);
      X1.load(q1 + 4);
      Y0.load(q2);
      Y1.load(q2 + 4);
      S0 = mul_add(X0, Y0, S0);
      S1 = mul_add(X1, Y1, S1);
      q1 += b;
      q2 += b;
    }
    // store result
    double s[4];
    S0.store(s);
    dp[bj + 0] = s[0];
    dp[bj + 1] = s[1];
    dp[bj + 2] = s[2];
    dp[bj + 3] = s[3];
    S1.store(s);
    dp[bj + 4] = s[0];
    dp[bj + 5] = s[1];
    dp[bj + 6] = s[2];
    dp[bj + 7] = s[3];
  }
}

// ! compute dot product of each column in Q1 with same column in Q2; vectorized for avx2
// template <typename MV>
// void dot_products_all_avx2_b8(std::vector<std::vector<double>> &dp, const MV &Q1, const MV &Q2)
// {
//   if (dp.size() != Q1.cols())
//     dp.resize(Q1.cols());
//   for (auto &v : dp)
//     if (dp.size() != Q1.cols())
//       dp.resize(Q1.cols());
//   if (Q1.rows() != Q2.rows())
//     throw std::invalid_argument("dot_products_blocked: number of rows does not match");
//   if (Q1.cols() != Q2.cols())
//     throw std::invalid_argument("dot_products_blocked: number of columns does not match");
//   std::size_t n = Q1.rows();
//   std::size_t m = Q1.cols();
//   std::size_t b = MV::blocksize;
//   Vec4d SS[8][2];
//   Vec4d S0, S1; // registers to store scalar products
//   Vec4d X0, X1;
//   Vec4d Y0, Y1;


// template <typename MV>
// void dot_products_all_avx2_b8(std::vector<std::vector<double>> &dp, const MV &Q1, const MV &Q2)
// {
//   if (Q1.rows() != Q2.rows())
//     throw std::invalid_argument("dot_products_blocked: number of rows does not match");
//   if (Q1.cols() != Q2.cols())
//     throw std::invalid_argument("dot_products_blocked: number of columns does not match");

//   std::size_t n = Q1.rows();
//   std::size_t m = Q1.cols();
//   std::size_t b = 8; // Block size

//   dp.resize(m, std::vector<double>(m, 0.0)); // Resize dp to match the result dimensions

//   Vec4d S0, S1; // registers to store scalar products
//   Vec4d X0, X1;
//   Vec4d Y0, Y1;

//   for (std::size_t bj1 = 0; bj1 < m; bj1 += b)
//   {
//     for (std::size_t bj2 = 0; bj2 < m; bj2 += b)
//     {
//       for (std::size_t i = 0; i < b; i += 4)
//       {
//         for (std::size_t j = 0; j < b; j += 4)
//         {
//           // clear summation variables
//           S0 = Vec4d(0.0);
//           S1 = Vec4d(0.0);

//           for (std::size_t k = 0; k < n; ++k)
//           {
//             // Load values from Q1 and Q2
//             X0.load(&(Q1(k, bj1 + i)));
//             X1.load(&(Q1(k, bj1 + i + 4)));
//             Y0.load(&(Q2(k, bj2 + j)));
//             Y1.load(&(Q2(k, bj2 + j + 4)));

//             // Perform the multiplication and add to the sum
//             S0 = mul_add(X0, Y0, S0);
//             S1 = mul_add(X1, Y1, S1);
//           }

//           // store results
//           double s[2][4];
//           S0.store(s[0]);
//           S1.store(s[1]);

//           for (std::size_t jj = 0; jj < 4; ++jj)
//           {
//             dp[bj1 + i + jj][bj2 + j] = s[0][jj];
//             dp[bj1 + i + jj][bj2 + j + 4] = s[1][jj];
//           }
//         }
//       }
//     }
//   }
// }

template <typename MV>
/** @brief Orthogonalize a given MultiVector, block version using AVX2 and block size 8
 *
 */
void orthonormalize_avx2_b8(MV &Q)
{
  using T = typename MV::value_type; // could be float or double
  if (MV::blocksize != 8)
    throw std::invalid_argument("orthonormalize_avx2_b8: blocksize must be 8");
  auto n = Q.rows();
  auto m = Q.cols();
  auto q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;

  for (std::size_t bk = 0; bk < m; bk += b) // loop over all columns of Q in blocks
  {
    // std::cout << "bk=" << bk << std::endl;

    std::size_t nbk = bk * n; // start of block of b columns in memory

    // allocate registers
    Vec4d SS[4][2]; // 8 Registers for the scalar products
    Vec4d QK, QJ0, QJ1;
    Vec4d minus1 = Vec4d(-1.0);

    // create memory for scalar products and initialize
    double s[b][b];
    for (std::size_t j = 0; j < b; ++j)
      for (std::size_t k = 0; k < b; ++k)
        s[k][j] = 0.0;

    // orthogonalization of diagonal block
    if (false)
    {
      // std::cout << "  diagonal block" << std::endl;
      for (std::size_t k = 0; k < b; ++k)
      {
        // compute scalar products of vectors in the block (upper triangle)
        // std::cout << "  scalar products" << std::endl;
        auto qbk = &(q[nbk]);
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
      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // first half of scalar products
      auto qbk = &(q[nbk]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbk);
        QJ1.load(qbk + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[0]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(qbk[1]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(qbk[2]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(qbk[3]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        qbk += b;
      }

      // store scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          (SS[k][j]).store(&(s[k][4 * j]));

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // second half of scalar products
      qbk = &(q[nbk]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbk);
        QJ1.load(qbk + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[4]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(qbk[5]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(qbk[6]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(qbk[7]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        qbk += b;
      }

      // store scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          (SS[k][j]).store(&(s[k + 4][4 * j]));

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

      // now we need to do the matrix vector product Q = VU
      Vec4d SUM0, SUM1;
      Vec4d VIK, UK0, UK1;
      auto vi = q + nbk;
      for (std::size_t i = 0; i < n; ++i)
      {
        // compute row i times U
        // clear sumation variables
        SUM0 = Vec4d(0.0);
        SUM1 = Vec4d(0.0);
        for (std::size_t k = 0; k < b; ++k)
        {
          UK0.load(&(U[k][0]));
          UK1.load(&(U[k][4]));
          VIK = Vec4d(vi[k]);
          SUM0 = mul_add(VIK, UK0, SUM0);
          SUM1 = mul_add(VIK, UK1, SUM1);
        }
        SUM0.store(vi);
        SUM1.store(vi + 4);
        vi += b;
      }
    }

    // now do projection for all the remaining blocks with this block
    for (std::size_t bj = bk + b; bj < m; bj += b)
    {
      // std::cout << "    bj=" << bj << std::endl;

      std::size_t nbj = bj * n; // start of block of j columns in memory

      // first half of vectors in block bk

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // compute scalar products
      auto qbk = &(q[nbk]);
      auto qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[0]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(qbk[1]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(qbk[2]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(qbk[3]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        qbk += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[0]); // element 0 in block bk
        QJ0 = mul_add(SS[0][0], QK, QJ0);
        QJ1 = mul_add(SS[0][1], QK, QJ1);
        QK = Vec4d(qbk[1]); // element 1 in block bk
        QJ0 = mul_add(SS[1][0], QK, QJ0);
        QJ1 = mul_add(SS[1][1], QK, QJ1);
        QK = Vec4d(qbk[2]); // element 2 in block bk
        QJ0 = mul_add(SS[2][0], QK, QJ0);
        QJ1 = mul_add(SS[2][1], QK, QJ1);
        QK = Vec4d(qbk[3]); // element 3 in block bk
        QJ0 = mul_add(SS[3][0], QK, QJ0);
        QJ1 = mul_add(SS[3][1], QK, QJ1);
        QJ0.store(qbj);
        QJ1.store(qbj + 4); // load 8 entries in block bj
        qbk += b;
        qbj += b;
      }

      // second half of vectors in block bk

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // compute scalar products
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[4]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(qbk[5]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(qbk[6]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(qbk[7]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        qbk += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[4]); // element 0 in block bk
        QJ0 = mul_add(SS[0][0], QK, QJ0);
        QJ1 = mul_add(SS[0][1], QK, QJ1);
        QK = Vec4d(qbk[5]); // element 1 in block bk
        QJ0 = mul_add(SS[1][0], QK, QJ0);
        QJ1 = mul_add(SS[1][1], QK, QJ1);
        QK = Vec4d(qbk[6]); // element 2 in block bk
        QJ0 = mul_add(SS[2][0], QK, QJ0);
        QJ1 = mul_add(SS[2][1], QK, QJ1);
        QK = Vec4d(qbk[7]); // element 3 in block bk
        QJ0 = mul_add(SS[3][0], QK, QJ0);
        QJ1 = mul_add(SS[3][1], QK, QJ1);
        QJ0.store(qbj);
        QJ1.store(qbj + 4); // load 8 entries in block bj
        qbk += b;
        qbj += b;
      }
    }
  }
}

template <typename MV>
/** @brief Orthogonalize a given MultiVector, block version using AVX2 and block size 8
 *
 */
void orthonormalize_avx2_b8_v2(MV &Q)
{
  using T = typename MV::value_type; // could be float or double
  if (MV::blocksize != 8)
    throw std::invalid_argument("orthonormalize_avx2_b8: blocksize must be 8");
  auto n = Q.rows();
  auto m = Q.cols();
  auto q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;

  for (std::size_t bk = 0; bk < m; bk += b) // loop over all columns of Q in blocks
  {
    // std::cout << "bk=" << bk << std::endl;

    std::size_t nbk = bk * n; // start of block of b columns in memory

    // allocate registers
    Vec4d SS[8][2]; // 8 Registers for the scalar products
    Vec4d QK, QJ0, QJ1;
    Vec4d minus1 = Vec4d(-1.0);

    // create memory for scalar products and initialize
    double s[b][b];
    for (std::size_t j = 0; j < b; ++j)
      for (std::size_t k = 0; k < b; ++k)
        s[k][j] = 0.0;

    // orthogonalization of diagonal block

    // clear scalar products
    for (std::size_t k = 0; k < 8; ++k)
      for (std::size_t j = 0; j < 2; ++j)
        SS[k][j] = Vec4d(0.0);

    // first half of scalar products
    auto qbk = &(q[nbk]);
    for (std::size_t i = 0; i < n; ++i)
    {
      QJ0.load(qbk);
      QJ1.load(qbk + 4);  // load 8 entries in block bj
      QK = Vec4d(qbk[0]); // element 0 in block bk
      SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
      SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
      QK = Vec4d(qbk[1]); // element 1 in block bk
      SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
      SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
      QK = Vec4d(qbk[2]); // element 2 in block bk
      SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
      SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
      QK = Vec4d(qbk[3]); // element 3 in block bk
      SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
      SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
      QK = Vec4d(qbk[4]); // element 0 in block bk
      SS[4][0] = mul_add(QK, QJ0, SS[4][0]);
      SS[4][1] = mul_add(QK, QJ1, SS[4][1]);
      QK = Vec4d(qbk[5]); // element 1 in block bk
      SS[5][0] = mul_add(QK, QJ0, SS[5][0]);
      SS[5][1] = mul_add(QK, QJ1, SS[5][1]);
      QK = Vec4d(qbk[6]); // element 2 in block bk
      SS[6][0] = mul_add(QK, QJ0, SS[6][0]);
      SS[6][1] = mul_add(QK, QJ1, SS[6][1]);
      QK = Vec4d(qbk[7]); // element 3 in block bk
      SS[7][0] = mul_add(QK, QJ0, SS[7][0]);
      SS[7][1] = mul_add(QK, QJ1, SS[7][1]);
      qbk += b;
    }

    // store scalar products
    for (std::size_t k = 0; k < 8; ++k)
      for (std::size_t j = 0; j < 2; ++j)
        (SS[k][j]).store(&(s[k][4 * j]));

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

    // now we need to do the matrix vector product Q = VU
    Vec4d SUM0, SUM1;
    Vec4d VIK, UK0, UK1;
    auto vi = q + nbk;
    for (std::size_t i = 0; i < n; ++i)
    {
      // compute row i times U
      // clear sumation variables
      SUM0 = Vec4d(0.0);
      SUM1 = Vec4d(0.0);
      for (std::size_t k = 0; k < b; ++k)
      {
        UK0.load(&(U[k][0]));
        UK1.load(&(U[k][4]));
        VIK = Vec4d(vi[k]);
        SUM0 = mul_add(VIK, UK0, SUM0);
        SUM1 = mul_add(VIK, UK1, SUM1);
      }
      SUM0.store(vi);
      SUM1.store(vi + 4);
      vi += b;
    }

    // now do projection for all the remaining blocks with this block
    for (std::size_t bj = bk + b; bj < m; bj += b)
    {
      // std::cout << "    bj=" << bj << std::endl;

      std::size_t nbj = bj * n; // start of block of j columns in memory

      // first half of vectors in block bk

      // clear scalar products
      for (std::size_t k = 0; k < 8; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // compute scalar products
      auto qbk = &(q[nbk]);
      auto qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[0]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(qbk[1]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(qbk[2]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(qbk[3]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        QK = Vec4d(qbk[4]); // element 0 in block bk
        SS[4][0] = mul_add(QK, QJ0, SS[4][0]);
        SS[4][1] = mul_add(QK, QJ1, SS[4][1]);
        QK = Vec4d(qbk[5]); // element 1 in block bk
        SS[5][0] = mul_add(QK, QJ0, SS[5][0]);
        SS[5][1] = mul_add(QK, QJ1, SS[5][1]);
        QK = Vec4d(qbk[6]); // element 2 in block bk
        SS[6][0] = mul_add(QK, QJ0, SS[6][0]);
        SS[6][1] = mul_add(QK, QJ1, SS[6][1]);
        QK = Vec4d(qbk[7]); // element 3 in block bk
        SS[7][0] = mul_add(QK, QJ0, SS[7][0]);
        SS[7][1] = mul_add(QK, QJ1, SS[7][1]);
        qbk += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 8; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[0]); // element 0 in block bk
        QJ0 = mul_add(SS[0][0], QK, QJ0);
        QJ1 = mul_add(SS[0][1], QK, QJ1);
        QK = Vec4d(qbk[1]); // element 1 in block bk
        QJ0 = mul_add(SS[1][0], QK, QJ0);
        QJ1 = mul_add(SS[1][1], QK, QJ1);
        QK = Vec4d(qbk[2]); // element 2 in block bk
        QJ0 = mul_add(SS[2][0], QK, QJ0);
        QJ1 = mul_add(SS[2][1], QK, QJ1);
        QK = Vec4d(qbk[3]); // element 3 in block bk
        QJ0 = mul_add(SS[3][0], QK, QJ0);
        QJ1 = mul_add(SS[3][1], QK, QJ1);
        QK = Vec4d(qbk[4]); // element 0 in block bk
        QJ0 = mul_add(SS[4][0], QK, QJ0);
        QJ1 = mul_add(SS[4][1], QK, QJ1);
        QK = Vec4d(qbk[5]); // element 1 in block bk
        QJ0 = mul_add(SS[5][0], QK, QJ0);
        QJ1 = mul_add(SS[5][1], QK, QJ1);
        QK = Vec4d(qbk[6]); // element 2 in block bk
        QJ0 = mul_add(SS[6][0], QK, QJ0);
        QJ1 = mul_add(SS[6][1], QK, QJ1);
        QK = Vec4d(qbk[7]); // element 3 in block bk
        QJ0 = mul_add(SS[7][0], QK, QJ0);
        QJ1 = mul_add(SS[7][1], QK, QJ1);
        QJ0.store(qbj);
        QJ1.store(qbj + 4); // load 8 entries in block bj
        qbk += b;
        qbj += b;
      }
    }
  }
}

template <typename ISTLM, typename MV>
/** @brief B-rthogonalize a given MultiVector, block version using AVX2 and block size 8
 *
 */
void B_orthonormalize_avx2_b8(const ISTLM &B, MV &Q)
{
  using block_type = typename ISTLM::block_type;
  const int br = block_type::rows;
  const int bc = block_type::cols;
  if (br != 1 || bc != 1)
    throw std::invalid_argument("B_orthonormalize_avx2_b8: only implemented for FieldMatrix<..,1,1>");
  using T = typename MV::value_type; // could be float or double
  if (MV::blocksize != 8)
    throw std::invalid_argument("orthonormalize_avx2_b8: blocksize must be 8");
  auto n = Q.rows();
  auto m = Q.cols();
  auto q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;

  // allocate space for one block of vectors to store B*vector
  T *p = new (std::align_val_t{64}) T[n * b]();

  // loop over all columns of Q in blocks
  for (std::size_t bk = 0; bk < m; bk += b)
  {
    std::size_t nbk = bk * n; // start of block of b columns in memory

    // now we need to compute B*(block of vectors) which we keep updated
    {
      Vec4d S0, S1;
      Vec4d X0, X1, AA;

      T *pi = p;        // start of where to store result
      T *qbk = q + nbk; // start of input block of columns
      for (auto row_iter = B.begin(); row_iter != B.end(); ++row_iter)
      {
        S0 = Vec4d(0.0);
        S1 = Vec4d(0.0);
        for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        {
          auto pj = qbk + col_iter.index() * b;
          AA = Vec4d(*col_iter);
          X0.load(pj);
          X1.load(pj + 4);
          S0 = mul_add(AA, X0, S0);
          S1 = mul_add(AA, X1, S1);
        }
        S0.store(pi);
        S1.store(pi + 4);
        pi += b;
      }
    }

    // create memory for scalar products and initialize
    double s[b][b];
    for (std::size_t k = 0; k < b; ++k)
      for (std::size_t j = 0; j < b; ++j)
        s[k][j] = 0.0;

    // allocate registers
    Vec4d SS[4][2]; // 8 Registers for the scalar products
    Vec4d QK, QJ0, QJ1;
    Vec4d minus1 = Vec4d(-1.0);

    if (false)
    {
      // orthogonalization of diagonal block
      // std::cout << "  diagonal block" << std::endl;
      for (std::size_t k = 0; k < b; ++k)
      {
        // compute B-scalar products of vectors in the block (upper triangle)
        // std::cout << "  scalar products" << std::endl;
        auto qbk = q + nbk; // start of block
        T *pi = p;          // start of block containing B*Q
        for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = k; j < b; ++j) // (m/b)*b*(b+1)/2   |  n*m*(b+1)/2
            s[k][j] += pi[k] * qbk[j];
          qbk += b;
          pi += b;
        }
        for (std::size_t j = k + 1; j < b; ++j)
          s[k][j] /= s[k][k];
        s[k][k] = 1.0 / std::sqrt(s[k][k]);

        // orthonormalize j>k with k and normalize
        qbk = q + nbk;
        for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = k + 1; j < b; ++j) // n*m*(b+1)/2
            qbk[j] -= s[k][j] * qbk[k];
          qbk[k] *= s[k][k]; // normalization
          qbk += b;
        }

        // update B*(block of vectors) in the same way
        pi = p;
        for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = k + 1; j < b; ++j) // n*m*(b+1)/2
            pi[j] -= s[k][j] * pi[k];
          pi[k] *= s[k][k]; // normalization
          pi += b;
        }
      }
    }
    else
    {
      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // first half of scalar products
      auto qbk = &(q[nbk]);
      T *pi = p; // start of block containing B*Q
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbk);
        QJ1.load(qbk + 4); // load 8 entries in block bj
        QK = Vec4d(pi[0]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(pi[1]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(pi[2]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(pi[3]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        qbk += b;
        pi += b;
      }

      // store scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          (SS[k][j]).store(&(s[k][4 * j]));

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // second half of scalar products
      qbk = &(q[nbk]);
      pi = p; // start of block containing B*Q
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbk);
        QJ1.load(qbk + 4); // load 8 entries in block bj
        QK = Vec4d(pi[4]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(pi[5]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(pi[6]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(pi[7]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        qbk += b;
        pi += b;
      }

      // store scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          (SS[k][j]).store(&(s[k + 4][4 * j]));

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

      // now we need to do the matrix vector product Q = VU
      Vec4d SUM0, SUM1;
      Vec4d VIK, UK0, UK1;
      auto vi = q + nbk;
      for (std::size_t i = 0; i < n; ++i)
      {
        // compute row i times U
        // clear sumation variables
        SUM0 = Vec4d(0.0);
        SUM1 = Vec4d(0.0);
        for (std::size_t k = 0; k < b; ++k)
        {
          UK0.load(&(U[k][0]));
          UK1.load(&(U[k][4]));
          VIK = Vec4d(vi[k]);
          SUM0 = mul_add(VIK, UK0, SUM0);
          SUM1 = mul_add(VIK, UK1, SUM1);
        }
        SUM0.store(vi);
        SUM1.store(vi + 4);
        vi += b;
      }

      // and we need to update the BVU
      pi = p;
      for (std::size_t i = 0; i < n; ++i)
      {
        // compute row i times U
        // clear sumation variables
        SUM0 = Vec4d(0.0);
        SUM1 = Vec4d(0.0);
        for (std::size_t k = 0; k < b; ++k)
        {
          UK0.load(&(U[k][0]));
          UK1.load(&(U[k][4]));
          VIK = Vec4d(pi[k]);
          SUM0 = mul_add(VIK, UK0, SUM0);
          SUM1 = mul_add(VIK, UK1, SUM1);
        }
        SUM0.store(pi);
        SUM1.store(pi + 4);
        pi += b;
      }
    }

    // now do projection for all the remaining blocks with this block
    for (std::size_t bj = bk + b; bj < m; bj += b)
    {
      std::size_t nbj = bj * n; // start of block of j columns in memory

      // first half of vectors in block bk
      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // compute scalar products
      auto pi = p;
      auto qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4); // load 8 entries in block bj
        QK = Vec4d(pi[0]); // element 0 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(pi[1]); // element 1 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(pi[2]); // element 2 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(pi[3]); // element 3 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        pi += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections; this stays the same
      auto qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[0]); // element 0 in block bk
        QJ0 = mul_add(SS[0][0], QK, QJ0);
        QJ1 = mul_add(SS[0][1], QK, QJ1);
        QK = Vec4d(qbk[1]); // element 1 in block bk
        QJ0 = mul_add(SS[1][0], QK, QJ0);
        QJ1 = mul_add(SS[1][1], QK, QJ1);
        QK = Vec4d(qbk[2]); // element 2 in block bk
        QJ0 = mul_add(SS[2][0], QK, QJ0);
        QJ1 = mul_add(SS[2][1], QK, QJ1);
        QK = Vec4d(qbk[3]); // element 3 in block bk
        QJ0 = mul_add(SS[3][0], QK, QJ0);
        QJ1 = mul_add(SS[3][1], QK, QJ1);
        QJ0.store(qbj);
        QJ1.store(qbj + 4); // load 8 entries in block bj
        qbk += b;
        qbj += b;
      }

      // second half of vectors in block bk

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = Vec4d(0.0);

      // compute scalar products
      pi = p;
      qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4); // load 8 entries in block bj
        QK = Vec4d(pi[4]); // element 4 in block bk
        SS[0][0] = mul_add(QK, QJ0, SS[0][0]);
        SS[0][1] = mul_add(QK, QJ1, SS[0][1]);
        QK = Vec4d(pi[5]); // element 5 in block bk
        SS[1][0] = mul_add(QK, QJ0, SS[1][0]);
        SS[1][1] = mul_add(QK, QJ1, SS[1][1]);
        QK = Vec4d(pi[6]); // element 6 in block bk
        SS[2][0] = mul_add(QK, QJ0, SS[2][0]);
        SS[2][1] = mul_add(QK, QJ1, SS[2][1]);
        QK = Vec4d(pi[7]); // element 7 in block bk
        SS[3][0] = mul_add(QK, QJ0, SS[3][0]);
        SS[3][1] = mul_add(QK, QJ1, SS[3][1]);
        pi += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 2; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0.load(qbj);
        QJ1.load(qbj + 4);  // load 8 entries in block bj
        QK = Vec4d(qbk[4]); // element 4 in block bk
        QJ0 = mul_add(SS[0][0], QK, QJ0);
        QJ1 = mul_add(SS[0][1], QK, QJ1);
        QK = Vec4d(qbk[5]); // element 5 in block bk
        QJ0 = mul_add(SS[1][0], QK, QJ0);
        QJ1 = mul_add(SS[1][1], QK, QJ1);
        QK = Vec4d(qbk[6]); // element 6 in block bk
        QJ0 = mul_add(SS[2][0], QK, QJ0);
        QJ1 = mul_add(SS[2][1], QK, QJ1);
        QK = Vec4d(qbk[7]); // element 7 in block bk
        QJ0 = mul_add(SS[3][0], QK, QJ0);
        QJ1 = mul_add(SS[3][1], QK, QJ1);
        QJ0.store(qbj);
        QJ1.store(qbj + 4); // load 8 entries in block bj
        qbk += b;
        qbj += b;
      }
    }
  }
  // free memory
  delete[] p;
}

/** @brief multiply sparse matrix with tall skinny matrix stored in a multivector
 *
 */
template <typename MV, typename ISTLM>
void matmul_sparse_tallskinny_avx2_b8(MV &Qout, const ISTLM &A, const MV &Qin)
{
  using block_type = typename ISTLM::block_type;
  const int br = block_type::rows;
  const int bc = block_type::cols;
  if (br != 1 || bc != 1)
    throw std::invalid_argument("matmul_sparse_tallskinny_avx2_b8: only implemented for FieldMatrix<..,1,1>");
  std::size_t n = Qin.rows();
  std::size_t m = Qin.cols();
  std::size_t block_size = MV::blocksize;
  auto pin = &(Qin(0, 0));
  auto pout = &(Qout(0, 0));
  Vec4d S0, S1;
  Vec4d X0, X1, AA;
  for (int bj = 0; bj < m; bj += block_size)
  {
    // x[i][j] = x[(j/b)*n*b + i*b + j%b]
    int nbj = n * bj;
    int I = n * bj;
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    {
      S0 = Vec4d(0.0);
      S1 = Vec4d(0.0);
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      {
        int J = nbj + col_iter.index() * block_size;
        AA = Vec4d(*col_iter);
        X0.load(&(pin[J]));
        X1.load(&(pin[J + 4]));
        S0 = mul_add(AA, X0, S0);
        S1 = mul_add(AA, X1, S1);
      }
      S0.store(&(pout[I]));
      S1.store(&(pout[I + 4]));
      I += block_size;
    }
  }
}

//! Apply inverse in factorized form; you may overwrite the input argument
template <typename MV, typename MAT>
void matmul_inverse_tallskinny_avx2_b8(MV &Qout, UMFPackFactorizedMatrix<MAT> &F, MV &Qin)
{
  // std::cout << "enter matmul_inverse_tallskinny_blocked" << std::endl;
  if (Qout.rows() != Qin.rows() || Qout.cols() != Qin.cols())
    throw std::invalid_argument("matmul_inverse_tallskinny_neon_b8: Qout/Qin size mismatch");
  if (F.n != Qin.rows() || F.n != Qout.rows())
    throw std::invalid_argument("matmul_inverse_tallskinny_neon_b8: Factorization does not match size of Qout/Qin");

  const int n = Qin.rows(); // rows in Q as well as size of square factorized matrix
  const int m = Qin.cols(); // columns in Q
  const int block_size = MV::blocksize;
  if (block_size != 8)
    throw std::invalid_argument("matmul_inverse_tallskinny_avx2_b8: block size must be 8");
  auto pin = &(Qin(0, 0));
  auto pout = &(Qout(0, 0));
  Vec4d S0, S1;
  Vec4d R0, R1;
  Vec4d X0, X1;
  Vec4d AA;

  // loop over all blocks of columns
  for (int bj = 0; bj < m; bj += block_size)
  {
    int nbj = n * bj;

    // combined row scaling and permutation
    // store result in Qout
    if (F.do_recip)
    {
      double *poutK = pout + nbj; // and a pointer to this entry
      for (int k = 0; k < n; ++k) // loop over rows
      {
        AA = Vec4d(F.Rs[F.P[k]]);
        double *pinI = pin + nbj + F.P[k] * block_size; // and a pointer to this entry

        X0.load(pinI);
        X1.load(pinI + 4);

        S0 = AA * X0;
        S1 = AA * X1;

        S0.store(poutK);
        S1.store(poutK + 4);

        poutK += block_size;
      }
    }
    else
    {
      double *poutK = pout + nbj; // and a pointer to this entry
      for (int k = 0; k < n; ++k) // loop over rows
      {
        AA = Vec4d(1.0 / F.Rs[F.P[k]]);
        double *pinI = pin + nbj + F.P[k] * block_size; // and a pointer to this entry

        X0.load(pinI);
        X1.load(pinI + 4);

        S0 = AA * X0;
        S1 = AA * X1;

        S0.store(poutK);
        S1.store(poutK + 4);

        poutK += block_size;
      }
    }

    // back solve L (in compressed row storage)
    // rhs are in Qout
    // store result in Qin
    {
      double *pinI = pin + nbj;   // and a pointer to this entry
      double *poutI = pout + nbj; // and a pointer to this entry
      double *pinJ;
      for (int i = 0; i < n; i++) // loop over all rows
      {
        // load right hand side
        S0.load(poutI);
        S1.load(poutI + 4);
        for (auto k = F.Lp[i]; k < F.Lp[i + 1] - 1; k++)
        {
          AA = Vec4d(F.Lx[k]);
          pinJ = pin + nbj + F.Lj[k] * block_size;
          X0.load(pinJ);
          X1.load(pinJ + 4);
          S0 = nmul_add(AA, X0, S0);
          S1 = nmul_add(AA, X1, S1);
        }
        S0.store(pinI);
        S1.store(pinI + 4);
        pinI += block_size;
        poutI += block_size;
      }
    }

    // back solve U (in compressed column storage)
    // input in Qin, output in Qin (so that we can store the permutation in Qout in the end)
    {
      double *pinJ = &(pin[nbj + (n - 1) * block_size]);
      double *pinI;
      double *poutK;
      for (int j = n - 1; j >= 0; j--)
      {
        AA = Vec4d(F.Ux[F.Up[j + 1] - 1]);
        R0.load(pinJ);
        R1.load(pinJ + 4);
        R0 = R0 / AA;
        R1 = R1 / AA;
        for (auto k = F.Up[j]; k < F.Up[j + 1] - 1; k++)
        {
          pinI = &(pin[nbj + F.Ui[k] * block_size]);
          AA = Vec4d(F.Ux[k]);
          X0.load(pinI);
          X1.load(pinI + 4);
          X0 = nmul_add(AA, R0, X0);
          X1 = nmul_add(AA, R1, X1);
          X0.store(pinI);
          X1.store(pinI + 4);
        }
        // store result at permuted index
        poutK = &(pout[nbj + F.Q[j] * block_size]);
        R0.store(poutK);
        R1.store(poutK + 4);
        pinJ -= block_size;
      }
    }
  }
}

#endif
