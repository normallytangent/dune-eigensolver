#ifndef Udune_eigensolver_kernels_neon_HH
#define Udune_eigensolver_kernels_neon_HH

#include "umfpacktools.hh"
#include <arm_neon.h>

//! compute dot product of each column in Q1 with same column in Q2; vectorized for neon
template <typename MV>
void dot_products_diagonal_neon_b8(std::vector<double> &dp, const MV &Q1, const MV &Q2)
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
  float64x2_t S0, S1, S2, S3; // registers to store scalar products
  float64x2_t X0, X1, X2, X3;
  float64x2_t Y0, Y1, Y2, Y3;

  for (std::size_t bj = 0; bj < m; bj += b)
  {
    // pointers to start of blocks
    auto q1 = &(Q1(0, bj));
    auto q2 = &(Q2(0, bj));

    // clear summation variable
    S0 = vmovq_n_f64(0.0);
    S1 = vmovq_n_f64(0.0);
    S2 = vmovq_n_f64(0.0);
    S3 = vmovq_n_f64(0.0);

    for (std::size_t i = 0; i < n; ++i)
    {
      X0 = vld1q_f64(q1);
      X1 = vld1q_f64(q1 + 2);
      X2 = vld1q_f64(q1 + 4);
      X3 = vld1q_f64(q1 + 6);
      Y0 = vld1q_f64(q2);
      Y1 = vld1q_f64(q2 + 2);
      Y2 = vld1q_f64(q2 + 4);
      Y3 = vld1q_f64(q2 + 6);
      S0 = vfmaq_f64(S0, X0, Y0);
      S1 = vfmaq_f64(S1, X1, Y1);
      S2 = vfmaq_f64(S2, X2, Y2);
      S3 = vfmaq_f64(S3, X3, Y3);
      q1 += b;
      q2 += b;
    }
    // store result
    double s[2];
    vst1q_f64(s, S0);
    dp[bj + 0] = s[0];
    dp[bj + 1] = s[1];
    vst1q_f64(s, S1);
    dp[bj + 2] = s[0];
    dp[bj + 3] = s[1];
    vst1q_f64(s, S2);
    dp[bj + 4] = s[0];
    dp[bj + 5] = s[1];
    vst1q_f64(s, S3);
    dp[bj + 6] = s[0];
    dp[bj + 7] = s[1];
  }
}

/** @brief Orthogonalize a given MultiVector, block version using arm neon and block size 8
 *
 */
template <typename MV>
void orthonormalize_neon_b8(MV &Q)
{
  using T = typename MV::value_type; // could be float or double
  if (MV::blocksize != 8)
    throw std::invalid_argument("orthonormalize_neon_b8: blocksize must be 8");
  auto n = Q.rows();
  auto m = Q.cols();
  auto q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;
  for (std::size_t bk = 0; bk < m; bk += b) // loop over all columns of Q in blocks
  {
    // std::cout << "bk=" << bk << std::endl;

    int nbk = bk * n; // start of block of b columns in memory

    // create memory for scalar products and initialize
    float64x2_t SS[4][4]; // 16 Registers for the scalar products
    float64x2_t minus1 = vmovq_n_f64(-1.0);
    float64x2_t QK, QJ0, QJ1, QJ2, QJ3;

    // orthogonalization of diagonal block
    // this is done like in the scalar version; for neon we could think about doing something here as well
    // std::cout << "  diagonal block" << std::endl;
    for (std::size_t k = 0; k < 4; ++k)
      for (std::size_t j = 0; j < 4; ++j)
        SS[k][j] = vmovq_n_f64(0.0);

    // compute first half of scalar products
    auto qbk = &(q[nbk]);
    for (std::size_t i = 0; i < n; ++i)
    {
      QJ0 = vld1q_f64(qbk);
      QJ1 = vld1q_f64(qbk + 2);
      QJ2 = vld1q_f64(qbk + 4);
      QJ3 = vld1q_f64(qbk + 6);
      QK = vmovq_n_f64(qbk[0]); // element 0 in block bk
      SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
      SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
      SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
      SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[1]); // element 1 in block bk
      SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
      SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
      SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
      SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[2]); // element 2 in block bk
      SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
      SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
      SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
      SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[3]); // element 3 in block bk
      SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
      SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
      SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
      SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
      qbk += b;
    }

    // store in s matrix
    double s[b][b];
    for (std::size_t k = 0; k < 4; ++k)
      for (std::size_t j = 0; j < 4; ++j)
        vst1q_f64(&(s[k][2 * j]), SS[k][j]);

    // clear scalar products
    for (std::size_t k = 0; k < 4; ++k)
      for (std::size_t j = 0; j < 4; ++j)
        SS[k][j] = vmovq_n_f64(0.0);

    // and now the second half of scalar products
    qbk = &(q[nbk]);
    for (std::size_t i = 0; i < n; ++i)
    {
      QJ0 = vld1q_f64(qbk);
      QJ1 = vld1q_f64(qbk + 2);
      QJ2 = vld1q_f64(qbk + 4);
      QJ3 = vld1q_f64(qbk + 6);
      QK = vmovq_n_f64(qbk[4]); // element 0 in block bk
      SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
      SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
      SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
      SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[5]); // element 1 in block bk
      SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
      SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
      SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
      SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[6]); // element 2 in block bk
      SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
      SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
      SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
      SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[7]); // element 3 in block bk
      SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
      SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
      SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
      SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
      qbk += b;
    }
    // store in scalar matrix
    for (std::size_t k = 0; k < 4; ++k)
      for (std::size_t j = 0; j < 4; ++j)
        vst1q_f64(&(s[k + 4][2 * j]), SS[k][j]);

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
    float64x2_t SUM[4];
    float64x2_t VIK, UK[4];
    auto vi = q + nbk;
    for (std::size_t i = 0; i < n; ++i)
    {
      // compute row i times U
      // clear sumation variables
      for (std::size_t j = 0; j < 4; ++j)
        SUM[j] = vmovq_n_f64(0.0);
      for (std::size_t k = 0; k < b; ++k)
      {
        UK[0] = vld1q_f64(&(U[k][0]));
        UK[1] = vld1q_f64(&(U[k][2]));
        UK[2] = vld1q_f64(&(U[k][4]));
        UK[3] = vld1q_f64(&(U[k][6]));
        VIK = vmovq_n_f64(vi[k]);
        SUM[0] = vfmaq_f64(SUM[0], VIK, UK[0]);
        SUM[1] = vfmaq_f64(SUM[1], VIK, UK[1]);
        SUM[2] = vfmaq_f64(SUM[2], VIK, UK[2]);
        SUM[3] = vfmaq_f64(SUM[3], VIK, UK[3]);
      }
      vst1q_f64(vi, SUM[0]);
      vst1q_f64(vi + 2, SUM[1]);
      vst1q_f64(vi + 4, SUM[2]);
      vst1q_f64(vi + 6, SUM[3]);
      vi += b;
    }

    // now do projection for all the remaining blocks with this block
    for (std::size_t bj = bk + b; bj < m; bj += b)
    {
      // std::cout << "    bj=" << bj << std::endl;

      std::size_t nbj = bj * n; // start of block of j columns in memory

      // first half of vectors in block bk

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = vmovq_n_f64(0.0);

      // compute scalar products
      auto qbk = &(q[nbk]);
      auto qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[0]); // element 0 in block bk
        SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
        SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
        SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
        SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[1]); // element 1 in block bk
        SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
        SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
        SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
        SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[2]); // element 2 in block bk
        SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
        SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
        SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
        SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[3]); // element 3 in block bk
        SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
        SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
        SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
        SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
        qbk += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[0]); // element 0 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[0][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[0][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[0][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[0][3], QK);
        QK = vmovq_n_f64(qbk[1]); // element 1 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[1][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[1][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[1][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[1][3], QK);
        QK = vmovq_n_f64(qbk[2]); // element 2 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[2][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[2][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[2][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[2][3], QK);
        QK = vmovq_n_f64(qbk[3]); // element 3 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[3][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[3][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[3][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[3][3], QK);
        vst1q_f64(qbj, QJ0);
        vst1q_f64(qbj + 2, QJ1);
        vst1q_f64(qbj + 4, QJ2);
        vst1q_f64(qbj + 6, QJ3);
        qbk += b;
        qbj += b;
      }

      // second half of vectors in block bk

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = vmovq_n_f64(0.0);

      // compute scalar products
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[4]); // element 0 in block bk
        SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
        SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
        SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
        SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[5]); // element 1 in block bk
        SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
        SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
        SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
        SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[6]); // element 2 in block bk
        SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
        SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
        SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
        SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[7]); // element 3 in block bk
        SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
        SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
        SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
        SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
        qbk += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[4]); // element 0 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[0][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[0][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[0][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[0][3], QK);
        QK = vmovq_n_f64(qbk[5]); // element 1 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[1][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[1][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[1][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[1][3], QK);
        QK = vmovq_n_f64(qbk[6]); // element 2 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[2][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[2][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[2][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[2][3], QK);
        QK = vmovq_n_f64(qbk[7]); // element 3 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[3][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[3][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[3][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[3][3], QK);
        vst1q_f64(qbj, QJ0);
        vst1q_f64(qbj + 2, QJ1);
        vst1q_f64(qbj + 4, QJ2);
        vst1q_f64(qbj + 6, QJ3);
        qbk += b;
        qbj += b;
      }
    }
  }
}


/** @brief Orthogonalize a given MultiVector, block version using arm neon and block size 8
 *
 */
template <typename MV>
void orthonormalize_neon_b8_v2(MV &Q)
{
  using T = typename MV::value_type; // could be float or double
  if (MV::blocksize != 8)
    throw std::invalid_argument("orthonormalize_neon_b8: blocksize must be 8");
  auto n = Q.rows();
  auto m = Q.cols();
  auto q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;
  for (std::size_t bk = 0; bk < m; bk += b) // loop over all columns of Q in blocks
  {
    // std::cout << "bk=" << bk << std::endl;

    int nbk = bk * n; // start of block of b columns in memory

    // create memory for scalar products and initialize
    float64x2_t SS[8][4]; // 32 Registers for the scalar products
    float64x2_t minus1 = vmovq_n_f64(-1.0);
    float64x2_t QK, QJ0, QJ1, QJ2, QJ3;

    // orthogonalization of diagonal block
    // this is done like in the scalar version; for neon we could think about doing something here as well
    // std::cout << "  diagonal block" << std::endl;
    for (std::size_t k = 0; k < 8; ++k)
      for (std::size_t j = 0; j < 4; ++j)
        SS[k][j] = vmovq_n_f64(0.0);

    auto qbk = &(q[nbk]);
    for (std::size_t i = 0; i < n; ++i)
    {
      QJ0 = vld1q_f64(qbk);
      QJ1 = vld1q_f64(qbk + 2);
      QJ2 = vld1q_f64(qbk + 4);
      QJ3 = vld1q_f64(qbk + 6);
      QK = vmovq_n_f64(qbk[0]); // element 0 in block bk
      SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
      SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
      SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
      SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[1]); // element 1 in block bk
      SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
      SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
      SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
      SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[2]); // element 2 in block bk
      SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
      SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
      SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
      SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[3]); // element 3 in block bk
      SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
      SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
      SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
      SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[4]); // element 0 in block bk
      SS[4][0] = vfmaq_f64(SS[4][0], QK, QJ0);
      SS[4][1] = vfmaq_f64(SS[4][1], QK, QJ1);
      SS[4][2] = vfmaq_f64(SS[4][2], QK, QJ2);
      SS[4][3] = vfmaq_f64(SS[4][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[5]); // element 1 in block bk
      SS[5][0] = vfmaq_f64(SS[5][0], QK, QJ0);
      SS[5][1] = vfmaq_f64(SS[5][1], QK, QJ1);
      SS[5][2] = vfmaq_f64(SS[5][2], QK, QJ2);
      SS[5][3] = vfmaq_f64(SS[5][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[6]); // element 2 in block bk
      SS[6][0] = vfmaq_f64(SS[6][0], QK, QJ0);
      SS[6][1] = vfmaq_f64(SS[6][1], QK, QJ1);
      SS[6][2] = vfmaq_f64(SS[6][2], QK, QJ2);
      SS[6][3] = vfmaq_f64(SS[6][3], QK, QJ3);
      QK = vmovq_n_f64(qbk[7]); // element 3 in block bk
      SS[7][0] = vfmaq_f64(SS[7][0], QK, QJ0);
      SS[7][1] = vfmaq_f64(SS[7][1], QK, QJ1);
      SS[7][2] = vfmaq_f64(SS[7][2], QK, QJ2);
      SS[7][3] = vfmaq_f64(SS[7][3], QK, QJ3);
      qbk += b;
    }

    // store in s matrix
    double s[b][b];
    for (std::size_t k = 0; k < 8; ++k)
      for (std::size_t j = 0; j < 4; ++j)
        vst1q_f64(&(s[k][2 * j]), SS[k][j]);

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
    float64x2_t SUM[4];
    float64x2_t VIK, UK[4];
    auto vi = q + nbk;
    for (std::size_t i = 0; i < n; ++i)
    {
      // compute row i times U
      // clear sumation variables
      for (std::size_t j = 0; j < 4; ++j)
        SUM[j] = vmovq_n_f64(0.0);
      for (std::size_t k = 0; k < b; ++k)
      {
        UK[0] = vld1q_f64(&(U[k][0]));
        UK[1] = vld1q_f64(&(U[k][2]));
        UK[2] = vld1q_f64(&(U[k][4]));
        UK[3] = vld1q_f64(&(U[k][6]));
        VIK = vmovq_n_f64(vi[k]);
        SUM[0] = vfmaq_f64(SUM[0], VIK, UK[0]);
        SUM[1] = vfmaq_f64(SUM[1], VIK, UK[1]);
        SUM[2] = vfmaq_f64(SUM[2], VIK, UK[2]);
        SUM[3] = vfmaq_f64(SUM[3], VIK, UK[3]);
      }
      vst1q_f64(vi, SUM[0]);
      vst1q_f64(vi + 2, SUM[1]);
      vst1q_f64(vi + 4, SUM[2]);
      vst1q_f64(vi + 6, SUM[3]);
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
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = vmovq_n_f64(0.0);

      // compute scalar products
      auto qbk = &(q[nbk]);
      auto qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[0]); // element 0 in block bk
        SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
        SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
        SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
        SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[1]); // element 1 in block bk
        SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
        SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
        SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
        SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[2]); // element 2 in block bk
        SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
        SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
        SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
        SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[3]); // element 3 in block bk
        SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
        SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
        SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
        SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[4]); // element 0 in block bk
        SS[4][0] = vfmaq_f64(SS[4][0], QK, QJ0);
        SS[4][1] = vfmaq_f64(SS[4][1], QK, QJ1);
        SS[4][2] = vfmaq_f64(SS[4][2], QK, QJ2);
        SS[4][3] = vfmaq_f64(SS[4][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[5]); // element 1 in block bk
        SS[5][0] = vfmaq_f64(SS[5][0], QK, QJ0);
        SS[5][1] = vfmaq_f64(SS[5][1], QK, QJ1);
        SS[5][2] = vfmaq_f64(SS[5][2], QK, QJ2);
        SS[5][3] = vfmaq_f64(SS[5][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[6]); // element 2 in block bk
        SS[6][0] = vfmaq_f64(SS[6][0], QK, QJ0);
        SS[6][1] = vfmaq_f64(SS[6][1], QK, QJ1);
        SS[6][2] = vfmaq_f64(SS[6][2], QK, QJ2);
        SS[6][3] = vfmaq_f64(SS[6][3], QK, QJ3);
        QK = vmovq_n_f64(qbk[7]); // element 3 in block bk
        SS[7][0] = vfmaq_f64(SS[7][0], QK, QJ0);
        SS[7][1] = vfmaq_f64(SS[7][1], QK, QJ1);
        SS[7][2] = vfmaq_f64(SS[7][2], QK, QJ2);
        SS[7][3] = vfmaq_f64(SS[7][3], QK, QJ3);
        qbk += b;
        qbj += b;
      }

      // the minus sign
      for (std::size_t k = 0; k < 8; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = &(q[nbk]);
      qbj = &(q[nbj]);
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[0]); // element 0 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[0][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[0][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[0][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[0][3], QK);
        QK = vmovq_n_f64(qbk[1]); // element 1 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[1][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[1][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[1][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[1][3], QK);
        QK = vmovq_n_f64(qbk[2]); // element 2 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[2][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[2][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[2][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[2][3], QK);
        QK = vmovq_n_f64(qbk[3]); // element 3 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[3][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[3][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[3][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[3][3], QK);
        QK = vmovq_n_f64(qbk[4]); // element 0 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[4][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[4][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[4][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[4][3], QK);
        QK = vmovq_n_f64(qbk[5]); // element 1 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[5][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[5][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[5][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[5][3], QK);
        QK = vmovq_n_f64(qbk[6]); // element 2 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[6][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[6][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[6][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[6][3], QK);
        QK = vmovq_n_f64(qbk[7]); // element 3 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[7][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[7][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[7][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[7][3], QK);
        vst1q_f64(qbj, QJ0);
        vst1q_f64(qbj + 2, QJ1);
        vst1q_f64(qbj + 4, QJ2);
        vst1q_f64(qbj + 6, QJ3);
        qbk += b;
        qbj += b;
      }
    }
  }
}

/** @brief Orthogonalize a given MultiVector, block version using arm neon and block size 8
 *
 */
template <typename ISTLM, typename MV>
double B_orthonormalize_neon_b8(const ISTLM &B, MV &Q)
{
  using block_type = typename ISTLM::block_type;
  const int br = block_type::rows;
  const int bc = block_type::cols;
  if (br != 1 || bc != 1)
    throw std::invalid_argument("B_orthonormalize_neon_b8: only implemented for FieldMatrix<..,1,1>");
  using T = typename MV::value_type; // could be float or double
  if (MV::blocksize != 8)
    throw std::invalid_argument("B_orthonormalize_neon_b8: blocksize must be 8");
  auto n = Q.rows();
  auto m = Q.cols();
  auto q = &(Q(0, 0)); // work on raw data
  const std::size_t b = MV::blocksize;
  double norm = 0.0;

  // allocate space for one block of vectors to store B*vector
  T *p = new (std::align_val_t{64}) T[n * b]();

  // loop over all columns of Q in blocks
  for (std::size_t bk = 0; bk < m; bk += b)
  {
    // std::cout << "bk=" << bk << std::endl;
    int nbk = bk * n; // start of block of b columns in memory

    // now we need to compute B*(block of vectors) which we keep updated
    {
      float64x2_t S0, S1, S2, S3;
      float64x2_t X0, X1, X2, X3;
      float64x2_t AA;

      T *pi = p;        // start of where to store result
      T *qbk = q + nbk; // start of input block of columns
      for (auto row_iter = B.begin(); row_iter != B.end(); ++row_iter)
      {
        S0 = vmovq_n_f64(0.0);
        S1 = vmovq_n_f64(0.0);
        S2 = vmovq_n_f64(0.0);
        S3 = vmovq_n_f64(0.0);
        for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
        {
          auto pj = qbk + col_iter.index() * b;
          AA = vmovq_n_f64(*col_iter); // broadcast load scalar entry in vector register
          X0 = vld1q_f64(pj);
          X1 = vld1q_f64(pj + 2);
          X2 = vld1q_f64(pj + 4);
          X3 = vld1q_f64(pj + 6);
          S0 = vfmaq_f64(S0, AA, X0);
          S1 = vfmaq_f64(S1, AA, X1);
          S2 = vfmaq_f64(S2, AA, X2);
          S3 = vfmaq_f64(S3, AA, X3);
        }
        vst1q_f64(pi, S0);
        vst1q_f64(pi + 2, S1);
        vst1q_f64(pi + 4, S2);
        vst1q_f64(pi + 6, S3);
        pi += b;
      }
    }

    // create memory for scalar products and initialize
    float64x2_t SS[4][4]; // 16 Registers for the scalar products
    for (std::size_t k = 0; k < 4; ++k)
      for (std::size_t j = 0; j < 4; ++j)
        SS[k][j] = vmovq_n_f64(0.0);
    double s[b][b];
    float64x2_t minus1 = vmovq_n_f64(-1.0);
    // allocate vector registers
    float64x2_t QK, QJ0, QJ1, QJ2, QJ3;

    // orthogonalization of diagonal block
    // this is done like in the scalar version;
    // std::cout << "  diagonal block" << std::endl;
    if (false)
    {
      // for (std::size_t k = 0; k < b; k += 2)
      // {
      //   float64x2_t QJ; // remaining columns
      //   double s[2];

      //   // start with the even row
      //   // compute B-scalar products of vectors in the block (upper triangle)
      //   for (std::size_t j = 0; j < 4; ++j)
      //     SS[0][j] = vmovq_n_f64(0.0); // clear scalar products; use only first row
      //   auto qbk = q + nbk;            // start of block
      //   T *pi = p;                     // start of block containing B*Q
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     QK = vmovq_n_f64(pi[k]); // scalar broadcast load of diagonal element
      //     QJ = vld1q_f64(qbk + k); // load two values
      //     SS[0][k / 2] = vfmaq_f64(SS[0][k / 2], QK, QJ);
      //     for (std::size_t j = k + 2; j < b; j += 2) // (m/b)*b*(b+1)/2   |  n*m*(b+1)/2
      //     {
      //       QJ = vld1q_f64(qbk + j);
      //       SS[0][j / 2] = vfmaq_f64(SS[0][j / 2], QK, QJ);
      //     }
      //     qbk += b;
      //     pi += b;
      //   }
      //   vst1q_f64(s, SS[0][k / 2]); // split vector register into two
      //   QK = vmovq_n_f64(s[0]);     // broadcast diagonal element
      //   s[1] /= s[0];
      //   s[0] = 1.0 / std::sqrt(s[0]);
      //   for (std::size_t j = k + 2; j < b; j += 2)
      //     SS[0][j / 2] = vdivq_f64(SS[0][j / 2], QK);

      //   // orthonormalize j>k with k and normalize
      //   qbk = &(q[nbk]);
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     QK = vmovq_n_f64(qbk[k]);                  // scalar broadcast load of diagonal element
      //     qbk[k + 1] -= s[1] * qbk[k];               // diagonal plus 1
      //     for (std::size_t j = k + 2; j < b; j += 2) // the remaining blocks of 2 in that row
      //     {
      //       QJ = vld1q_f64(qbk + j);
      //       QJ = vfmsq_f64(QJ, SS[0][j / 2], QK);
      //       vst1q_f64(qbk + j, QJ);
      //     }
      //     qbk[k] *= s[0]; // normalization of diagonal
      //     qbk += b;
      //   }

      //   // update B*Q store in p array
      //   pi = p;
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     QK = vmovq_n_f64(pi[k]);                   // scalar broadcast load of diagonal element
      //     pi[k + 1] -= s[1] * pi[k];                 // diagonal plus 1
      //     for (std::size_t j = k + 2; j < b; j += 2) // the remaining blocks of 2 in that row
      //     {
      //       QJ = vld1q_f64(pi + j);
      //       QJ = vfmsq_f64(QJ, SS[0][j / 2], QK);
      //       vst1q_f64(pi + j, QJ);
      //     }
      //     pi[k] *= s[0]; // normalization of diagonal
      //     pi += b;
      //   }

      //   // and now the odd row
      //   // compute scalar products of vectors in the block (upper triangle)
      //   for (std::size_t j = 0; j < 4; ++j)
      //     SS[0][j] = vmovq_n_f64(0.0); // clear scalar products
      //   s[0] = s[1] = 0.0;
      //   qbk = q + nbk;
      //   pi = p;
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     QK = vmovq_n_f64(pi[k + 1]);               // scalar broadcast load of diagonal element
      //     s[0] += pi[k + 1] * qbk[k + 1];            // norm computation
      //     for (std::size_t j = k + 2; j < b; j += 2) // (m/b)*b*(b+1)/2   |  n*m*(b+1)/2
      //     {
      //       QJ = vld1q_f64(qbk + j);
      //       SS[0][j / 2] = vfmaq_f64(SS[0][j / 2], QK, QJ);
      //     }
      //     qbk += b;
      //     pi += b;
      //   }
      //   QK = vmovq_n_f64(s[0]); // broadcast diagonal element
      //   s[0] = 1.0 / std::sqrt(s[0]);
      //   for (std::size_t j = k + 2; j < b; j += 2)
      //     SS[0][j / 2] = vdivq_f64(SS[0][j / 2], QK);

      //   // orthonormalize j>k with k and normalize
      //   qbk = &(q[nbk]);
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     QK = vmovq_n_f64(qbk[k + 1]);              // scalar broadcast load of diagonal element
      //     for (std::size_t j = k + 2; j < b; j += 2) // the remaining blocks of 2 in that row
      //     {
      //       QJ = vld1q_f64(qbk + j);
      //       QJ = vfmsq_f64(QJ, SS[0][j / 2], QK);
      //       vst1q_f64(qbk + j, QJ);
      //     }
      //     qbk[k + 1] *= s[0]; // normalization of diagonal
      //     qbk += b;
      //   }

      //   // update B*Q accordingly
      //   pi = p;
      //   for (std::size_t i = 0; i < n; ++i)
      //   {
      //     QK = vmovq_n_f64(pi[k + 1]);               // scalar broadcast load of diagonal element
      //     for (std::size_t j = k + 2; j < b; j += 2) // the remaining blocks of 2 in that row
      //     {
      //       QJ = vld1q_f64(pi + j);
      //       QJ = vfmsq_f64(QJ, SS[0][j / 2], QK);
      //       vst1q_f64(pi + j, QJ);
      //     }
      //     pi[k + 1] *= s[0]; // normalization of diagonal
      //     pi += b;
      //   }
      // }
    }
    else
    {
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = vmovq_n_f64(0.0);

      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = 0; j < b; ++j)
          s[k][j] = 0.0;

      // compute first half of scalar products
      auto qbk = q + nbk;
      T *pi = p; // start of block containing B*Q
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbk);
        QJ1 = vld1q_f64(qbk + 2);
        QJ2 = vld1q_f64(qbk + 4);
        QJ3 = vld1q_f64(qbk + 6);
        QK = vmovq_n_f64(pi[0]); // element 0 in block bk
        SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
        SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
        SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
        SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
        QK = vmovq_n_f64(pi[1]); // element 1 in block bk
        SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
        SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
        SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
        SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
        QK = vmovq_n_f64(pi[2]); // element 2 in block bk
        SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
        SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
        SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
        SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
        QK = vmovq_n_f64(pi[3]); // element 3 in block bk
        SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
        SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
        SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
        SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
        qbk += b;
        pi += b;
      }

      // store in s matrix
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          vst1q_f64(&(s[k][2 * j]), SS[k][j]);

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = vmovq_n_f64(0.0);

      // and now the second half of scalar products
      qbk = &(q[nbk]);
      pi = p; // start of block containing B*Q
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbk);
        QJ1 = vld1q_f64(qbk + 2);
        QJ2 = vld1q_f64(qbk + 4);
        QJ3 = vld1q_f64(qbk + 6);
        QK = vmovq_n_f64(pi[4]); // element 0 in block bk
        SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
        SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
        SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
        SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
        QK = vmovq_n_f64(pi[5]); // element 1 in block bk
        SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
        SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
        SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
        SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
        QK = vmovq_n_f64(pi[6]); // element 2 in block bk
        SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
        SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
        SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
        SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
        QK = vmovq_n_f64(pi[7]); // element 3 in block bk
        SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
        SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
        SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
        SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
        qbk += b;
        pi += b;
      }

      // store in s matrix
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          vst1q_f64(&(s[k + 4][2 * j]), SS[k][j]);

      // update norm
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

      // now we need to do the matrix vector product Q = VU
      float64x2_t SUM[4];
      float64x2_t VIK, UK[4];
      auto vi = q + nbk;
      for (std::size_t i = 0; i < n; ++i)
      {
        // compute row i times U
        // clear sumation variables
        for (std::size_t j = 0; j < 4; ++j)
          SUM[j] = vmovq_n_f64(0.0);
        for (std::size_t k = 0; k < b; ++k)
        {
          UK[0] = vld1q_f64(&(U[k][0]));
          UK[1] = vld1q_f64(&(U[k][2]));
          UK[2] = vld1q_f64(&(U[k][4]));
          UK[3] = vld1q_f64(&(U[k][6]));
          VIK = vmovq_n_f64(vi[k]);
          SUM[0] = vfmaq_f64(SUM[0], VIK, UK[0]);
          SUM[1] = vfmaq_f64(SUM[1], VIK, UK[1]);
          SUM[2] = vfmaq_f64(SUM[2], VIK, UK[2]);
          SUM[3] = vfmaq_f64(SUM[3], VIK, UK[3]);
        }
        vst1q_f64(vi, SUM[0]);
        vst1q_f64(vi + 2, SUM[1]);
        vst1q_f64(vi + 4, SUM[2]);
        vst1q_f64(vi + 6, SUM[3]);
        vi += b;
      }

      // finally we need to update the B*vector
      pi = p;
      for (std::size_t i = 0; i < n; ++i)
      {
        // compute row i times U
        // clear sumation variables
        for (std::size_t j = 0; j < 4; ++j)
          SUM[j] = vmovq_n_f64(0.0);
        for (std::size_t k = 0; k < b; ++k)
        {
          UK[0] = vld1q_f64(&(U[k][0]));
          UK[1] = vld1q_f64(&(U[k][2]));
          UK[2] = vld1q_f64(&(U[k][4]));
          UK[3] = vld1q_f64(&(U[k][6]));
          VIK = vmovq_n_f64(pi[k]);
          SUM[0] = vfmaq_f64(SUM[0], VIK, UK[0]);
          SUM[1] = vfmaq_f64(SUM[1], VIK, UK[1]);
          SUM[2] = vfmaq_f64(SUM[2], VIK, UK[2]);
          SUM[3] = vfmaq_f64(SUM[3], VIK, UK[3]);
        }
        vst1q_f64(pi, SUM[0]);
        vst1q_f64(pi + 2, SUM[1]);
        vst1q_f64(pi + 4, SUM[2]);
        vst1q_f64(pi + 6, SUM[3]);
        pi += b;
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
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = vmovq_n_f64(0.0);

      // compute scalar products
      auto pi = p;
      auto qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(pi[0]); // element 0 in block bk
        SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
        SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
        SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
        SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
        QK = vmovq_n_f64(pi[1]); // element 1 in block bk
        SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
        SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
        SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
        SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
        QK = vmovq_n_f64(pi[2]); // element 2 in block bk
        SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
        SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
        SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
        SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
        QK = vmovq_n_f64(pi[3]); // element 3 in block bk
        SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
        SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
        SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
        SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
        pi += b;
        qbj += b;
      }

      // store in s matrix
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          vst1q_f64(&(s[k][2 * j]), SS[k][j]);

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      auto qbk = q + nbk;
      qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[0]); // element 0 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[0][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[0][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[0][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[0][3], QK);
        QK = vmovq_n_f64(qbk[1]); // element 1 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[1][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[1][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[1][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[1][3], QK);
        QK = vmovq_n_f64(qbk[2]); // element 2 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[2][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[2][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[2][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[2][3], QK);
        QK = vmovq_n_f64(qbk[3]); // element 3 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[3][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[3][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[3][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[3][3], QK);
        vst1q_f64(qbj, QJ0);
        vst1q_f64(qbj + 2, QJ1);
        vst1q_f64(qbj + 4, QJ2);
        vst1q_f64(qbj + 6, QJ3);
        qbk += b;
        qbj += b;
      }

      // second half of vectors in block bk

      // clear scalar products
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = vmovq_n_f64(0.0);

      // compute scalar products
      pi = p;
      qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(pi[4]); // element 0 in block bk
        SS[0][0] = vfmaq_f64(SS[0][0], QK, QJ0);
        SS[0][1] = vfmaq_f64(SS[0][1], QK, QJ1);
        SS[0][2] = vfmaq_f64(SS[0][2], QK, QJ2);
        SS[0][3] = vfmaq_f64(SS[0][3], QK, QJ3);
        QK = vmovq_n_f64(pi[5]); // element 1 in block bk
        SS[1][0] = vfmaq_f64(SS[1][0], QK, QJ0);
        SS[1][1] = vfmaq_f64(SS[1][1], QK, QJ1);
        SS[1][2] = vfmaq_f64(SS[1][2], QK, QJ2);
        SS[1][3] = vfmaq_f64(SS[1][3], QK, QJ3);
        QK = vmovq_n_f64(pi[6]); // element 2 in block bk
        SS[2][0] = vfmaq_f64(SS[2][0], QK, QJ0);
        SS[2][1] = vfmaq_f64(SS[2][1], QK, QJ1);
        SS[2][2] = vfmaq_f64(SS[2][2], QK, QJ2);
        SS[2][3] = vfmaq_f64(SS[2][3], QK, QJ3);
        QK = vmovq_n_f64(pi[7]); // element 3 in block bk
        SS[3][0] = vfmaq_f64(SS[3][0], QK, QJ0);
        SS[3][1] = vfmaq_f64(SS[3][1], QK, QJ1);
        SS[3][2] = vfmaq_f64(SS[3][2], QK, QJ2);
        SS[3][3] = vfmaq_f64(SS[3][3], QK, QJ3);
        pi += b;
        qbj += b;
      }

      // store in s matrix
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          vst1q_f64(&(s[k + 4][2 * j]), SS[k][j]);

      // update norm
      for (std::size_t k = 0; k < b; ++k)
        for (std::size_t j = 0; j < b; ++j)
          norm = std::max(norm, s[k][j]);

      // the minus sign
      for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
          SS[k][j] = minus1 * SS[k][j];

      // do projections
      qbk = q + nbk;
      qbj = q + nbj;
      for (std::size_t i = 0; i < n; ++i)
      {
        QJ0 = vld1q_f64(qbj);
        QJ1 = vld1q_f64(qbj + 2);
        QJ2 = vld1q_f64(qbj + 4);
        QJ3 = vld1q_f64(qbj + 6);
        QK = vmovq_n_f64(qbk[4]); // element 0 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[0][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[0][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[0][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[0][3], QK);
        QK = vmovq_n_f64(qbk[5]); // element 1 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[1][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[1][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[1][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[1][3], QK);
        QK = vmovq_n_f64(qbk[6]); // element 2 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[2][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[2][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[2][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[2][3], QK);
        QK = vmovq_n_f64(qbk[7]); // element 3 in block bk
        QJ0 = vfmaq_f64(QJ0, SS[3][0], QK);
        QJ1 = vfmaq_f64(QJ1, SS[3][1], QK);
        QJ2 = vfmaq_f64(QJ2, SS[3][2], QK);
        QJ3 = vfmaq_f64(QJ3, SS[3][3], QK);
        vst1q_f64(qbj, QJ0);
        vst1q_f64(qbj + 2, QJ1);
        vst1q_f64(qbj + 4, QJ2);
        vst1q_f64(qbj + 6, QJ3);
        qbk += b;
        qbj += b;
      }
    }
  }
  // free memory
  delete[] p;

  return norm;
}

/** @brief multiply sparse matrix with tall skinny matrix stored in a multivector
 *
 */
template <typename MV, typename ISTLM>
void matmul_sparse_tallskinny_neon_b8(MV &Qout, const ISTLM &A, const MV &Qin)
{
  using block_type = typename ISTLM::block_type;
  const int br = block_type::rows;
  const int bc = block_type::cols;
  if (br != 1 || bc != 1)
    throw std::invalid_argument("matmul_sparse_tallskinny_neon_b8: only implemented for FieldMatrix<..,1,1>");
  std::size_t n = Qin.rows();
  std::size_t m = Qin.cols();
  std::size_t block_size = MV::blocksize;
  auto pin = &(Qin(0, 0));
  auto pout = &(Qout(0, 0));
  float64x2_t S0, S1, S2, S3;
  float64x2_t X0, X1, X2, X3;
  float64x2_t AA;
  for (int bj = 0; bj < m; bj += block_size)
  {
    int nbj = n * bj;          // index of first entry in this block of vectors
    double *pi = &(pout[nbj]); // and a pointer to this entry
    for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
    {
      S0 = vmovq_n_f64(0.0);
      S1 = vmovq_n_f64(0.0);
      S2 = vmovq_n_f64(0.0);
      S3 = vmovq_n_f64(0.0);
      for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
      {
        int J = nbj + col_iter.index() * block_size;
        auto pj = pin + J;
        AA = vmovq_n_f64(*col_iter); // broadcast load scalar entry in vector register
        X0 = vld1q_f64(pj);
        X1 = vld1q_f64(pj + 2);
        X2 = vld1q_f64(pj + 4);
        X3 = vld1q_f64(pj + 6);
        S0 = vfmaq_f64(S0, AA, X0);
        S1 = vfmaq_f64(S1, AA, X1);
        S2 = vfmaq_f64(S2, AA, X2);
        S3 = vfmaq_f64(S3, AA, X3);
      }
      vst1q_f64(pi, S0);
      vst1q_f64(pi + 2, S1);
      vst1q_f64(pi + 4, S2);
      vst1q_f64(pi + 6, S3);
      pi += block_size;
    }
  }
}

//! Apply inverse in factorized form; you may overwrite the input argument
template <typename MV, typename MAT>
void matmul_inverse_tallskinny_neon_b8(MV &Qout, UMFPackFactorizedMatrix<MAT> &F, MV &Qin)
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
    throw std::invalid_argument("matmul_inverse_tallskinny_neon_b8: block size must be 8");
  auto pin = &(Qin(0, 0));
  auto pout = &(Qout(0, 0));
  float64x2_t S0, S1, S2, S3;
  float64x2_t R0, R1, R2, R3;
  float64x2_t X0, X1, X2, X3;
  float64x2_t AA;

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
        AA = vmovq_n_f64(F.Rs[F.P[k]]);
        double *pinI = pin + nbj + F.P[k] * block_size; // and a pointer to this entry

        X0 = vld1q_f64(pinI);
        X1 = vld1q_f64(pinI + 2);
        X2 = vld1q_f64(pinI + 4);
        X3 = vld1q_f64(pinI + 6);

        S0 = vmulq_f64(AA, X0);
        S1 = vmulq_f64(AA, X1);
        S2 = vmulq_f64(AA, X2);
        S3 = vmulq_f64(AA, X3);

        vst1q_f64(poutK, S0);
        vst1q_f64(poutK + 2, S1);
        vst1q_f64(poutK + 4, S2);
        vst1q_f64(poutK + 6, S3);

        poutK += block_size;
      }
    }
    else
    {
      double *poutK = pout + nbj; // and a pointer to this entry
      for (int k = 0; k < n; ++k) // loop over rows
      {
        AA = vmovq_n_f64(1.0 / F.Rs[F.P[k]]);
        double *pinI = pin + nbj + F.P[k] * block_size; // and a pointer to this entry

        X0 = vld1q_f64(pinI);
        X1 = vld1q_f64(pinI + 2);
        X2 = vld1q_f64(pinI + 4);
        X3 = vld1q_f64(pinI + 6);

        S0 = vmulq_f64(AA, X0);
        S1 = vmulq_f64(AA, X1);
        S2 = vmulq_f64(AA, X2);
        S3 = vmulq_f64(AA, X3);

        vst1q_f64(poutK, S0);
        vst1q_f64(poutK + 2, S1);
        vst1q_f64(poutK + 4, S2);
        vst1q_f64(poutK + 6, S3);

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
        S0 = vld1q_f64(poutI);
        S1 = vld1q_f64(poutI + 2);
        S2 = vld1q_f64(poutI + 4);
        S3 = vld1q_f64(poutI + 6);
        for (auto k = F.Lp[i]; k < F.Lp[i + 1] - 1; k++)
        {
          AA = vmovq_n_f64(F.Lx[k]);
          pinJ = pin + nbj + F.Lj[k] * block_size;
          X0 = vld1q_f64(pinJ);
          X1 = vld1q_f64(pinJ + 2);
          X2 = vld1q_f64(pinJ + 4);
          X3 = vld1q_f64(pinJ + 6);
          S0 = vfmsq_f64(S0, AA, X0);
          S1 = vfmsq_f64(S1, AA, X1);
          S2 = vfmsq_f64(S2, AA, X2);
          S3 = vfmsq_f64(S3, AA, X3);
        }
        vst1q_f64(pinI, S0);
        vst1q_f64(pinI + 2, S1);
        vst1q_f64(pinI + 4, S2);
        vst1q_f64(pinI + 6, S3);
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
        AA = vmovq_n_f64(F.Ux[F.Up[j + 1] - 1]);
        R0 = vld1q_f64(pinJ);
        R1 = vld1q_f64(pinJ + 2);
        R2 = vld1q_f64(pinJ + 4);
        R3 = vld1q_f64(pinJ + 6);
        R0 = vdivq_f64(R0, AA);
        R1 = vdivq_f64(R1, AA);
        R2 = vdivq_f64(R2, AA);
        R3 = vdivq_f64(R3, AA);
        for (auto k = F.Up[j]; k < F.Up[j + 1] - 1; k++)
        {
          pinI = &(pin[nbj + F.Ui[k] * block_size]);
          AA = vmovq_n_f64(F.Ux[k]);
          X0 = vld1q_f64(pinI);
          X1 = vld1q_f64(pinI + 2);
          X2 = vld1q_f64(pinI + 4);
          X3 = vld1q_f64(pinI + 6);
          X0 = vfmsq_f64(X0, AA, R0);
          X1 = vfmsq_f64(X1, AA, R1);
          X2 = vfmsq_f64(X2, AA, R2);
          X3 = vfmsq_f64(X3, AA, R3);
          vst1q_f64(pinI, X0);
          vst1q_f64(pinI + 2, X1);
          vst1q_f64(pinI + 4, X2);
          vst1q_f64(pinI + 6, X3);
        }
        // store result at permuted index
        poutK = &(pout[nbj + F.Q[j] * block_size]);
        vst1q_f64(poutK, R0);
        vst1q_f64(poutK + 2, R1);
        vst1q_f64(poutK + 4, R2);
        vst1q_f64(poutK + 6, R3);
        pinJ -= block_size;
      }
    }
  }
}

#endif
