#ifndef Udune_eigensolver_multivector_HH
#define Udune_eigensolver_multivector_HH

/**
 * @brief Store a tall and skinny matrix in block column-major order suitable for vectorization
 *
 * A typical access pattern would look like this:
 *
 *   for (std::size_t bj=0; bj<Q.cols(); bj+=MultiVector<double,4>::blocksize)
 *     for (std::size_t i=0; i<Q.rows(); ++i)
 *       for (std::size_t j=0; j<MultiVector<double,4>::blocksize; ++j)
 *         std::cout << "i=" << i << " j=" << bj+j << " " << &(Q(i,bj+j)) << std::endl;
 *
 * @tparam T a floating point type
 * @tparam b block size
 */
template <typename T, std::size_t b = 8>
class MultiVector
{
  T *p;          // pointer to data
  std::size_t n; // number of rows
  std::size_t m; // number of columns

public:
  //! export compile-time known block size
  static const std::size_t blocksize = b;

  //! export element type
  using value_type = T;

  //! constructor without arguments
  MultiVector()
  {
    // std::cout << "constructor w/o arguments this=" << this << std::endl;
    n = 0;
    m = 0;
    p = nullptr;
    // std::cout << "  p=" << p << std::endl;
  }

  //! construct nxm matrix
  /**
   * The number of columns must be a multiple of the block size
   */
  MultiVector(std::size_t n_, std::size_t m_)
  {
    // std::cout << "constructor this=" << this << " n=" << n_ << " m=" << m_ << std::endl;
    if (m_ % b != 0)
      throw std::invalid_argument("number of cols must be a multiple of block size");
    n = n_;
    m = m_;
    p = new (std::align_val_t{64}) T[n * m]();
    // std::cout << "  p=" << p << std::endl;
  }

  // copy constructor with deep copy
  MultiVector(const MultiVector<T, b> &o)
  {
    // std::cout << "copy constructor this=" << this << " o=" << &o << std::endl;
    n = o.n;
    m = o.m;
    p = new (std::align_val_t{64}) T[n * m]();
    for (std::size_t i = 0; i < n * m; ++i)
      p[i] = o.p[i];
    // std::cout << "  p=" << p << std::endl;
  }

  // move constructor
  MultiVector(MultiVector<T, b> &&o)
  {
    // std::cout << "move constructor this=" << this << " o=" << &o << std::endl;
    n = o.n;
    m = o.m;
    p = o.p; // steal pointer
    o.n = 0; // clear other
    o.m = 0;
    o.p = nullptr;
    // std::cout << "  p=" << p << " o.p=" << o.p << std::endl;
  }

  //! deallocate data
  ~MultiVector()
  {
    // std::cout << "destructor this=" << this << " p=" << p << std::endl;
    if (p != nullptr)
      delete[] p;
  }

  //! assignment operator with deep copy
  MultiVector<T, b> &operator=(const MultiVector<T, b> &o)
  {
    // std::cout << "assigment this=" << this << " o=" << &o << std::endl;
    if (this != &o)
    {
      if (n != o.n || m != o.m)
      {
        if (p != nullptr)
          delete[] p;
        p = new (std::align_val_t{64}) T[o.n * o.m]();
      }
      n = o.n;
      m = o.m;
      for (std::size_t i = 0; i < n * m; ++i)
        p[i] = o.p[i];
    }
    return *this;
  }

  //! move assignment operator
  MultiVector<T, b> &operator=(MultiVector<T, b> &&o)
  {
    // std::cout << "move assigment this=" << this << " o=" << &o << std::endl;
    if (this != &o)
    {
      if (p != nullptr)
        std::cout << "  deleting p=" << p << std::endl;
      if (p != nullptr)
        delete[] p; // delete own data
      n = o.n;      // steal other objects data
      m = o.m;
      p = o.p;
      o.n = 0;
      o.m = 0;
      o.p = nullptr;
    }
    return *this;
  }

  //! array access
  T &operator()(std::size_t i, std::size_t j)
  {
    return p[((j / blocksize) * n + i) * blocksize + (j % blocksize)];
  }

  //! const array access
  const T &operator()(std::size_t i, std::size_t j) const
  {
    // std::cout << ((j / blocksize) * n + i) * blocksize + (j % blocksize) << std::endl;
    return p[((j / blocksize) * n + i) * blocksize + (j % blocksize)];
  }

  //! get number of rows
  std::size_t rows() const { return n; }

  // get number of columns
  std::size_t cols() const { return m; }
};

template <typename T, std::size_t b>
inline std::ostream &operator<<(std::ostream &s, const MultiVector<T, b> &Q)
{
  s << "rows=" << Q.rows() << " cols=" << Q.cols() << std::endl;
  for (std::size_t i = 0; i < Q.rows(); i++)
  {
    s << std::setw(5) << i << " ";
    for (std::size_t j = 0; j < Q.cols(); j++)
      s
          << std::setw(12)
          << std::scientific
          << std::showpoint
          << std::setprecision(4)
          << Q(i, j);
    s << std::endl;
  }
  return s;
}

//********************************************************
// some useful output functions
//********************************************************

void show(const std::vector<std::vector<double>> &s)
{
  std::cout << "     ";
  for (std::size_t j = 0; j < s[0].size(); j++)
    std::cout << std::setw(12) << j;
  std::cout << std::endl;
  for (std::size_t i = 0; i < s.size(); i++)
  {
    std::cout << std::setw(5) << i;
    for (std::size_t j = 0; j < s[i].size(); j++)
      std::cout
          << std::setw(12)
          << std::scientific
          << std::showpoint
          << std::setprecision(4)
          << s[i][j];
    std::cout << std::endl;
  }
}

void show(const std::vector<double> &s)
{
  for (std::size_t j = 0; j < s.size(); j++)
    std::cout
        << std::setw(12)
        << std::scientific
        << std::showpoint
        << std::setprecision(4)
        << s[j];
  std::cout << std::endl;
}

void show(double* p, int n, int m)
{
  std::cout << "     ";
  for (std::size_t j = 0; j < m; j++)
    std::cout << std::setw(12) << j;
  std::cout << std::endl;
  for (std::size_t i = 0; i < n; i++)
  {
    std::cout << std::setw(5) << i;
    for (std::size_t j = 0; j < m; j++)
      std::cout
          << std::setw(12)
          << std::scientific
          << std::showpoint
          << std::setprecision(4)
          << p[i*m+j]; // row major access
    std::cout << std::endl;
  }
}

void show(double* p, int n)
{
  for (std::size_t j = 0; j < n; j++)
    std::cout
        << std::setw(12)
        << std::scientific
        << std::showpoint
        << std::setprecision(4)
        << p[j];
  std::cout << std::endl;
}


#endif
