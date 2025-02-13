#ifndef Udune_eigensolver_umfpacktools_HH
#define Udune_eigensolver_umfpacktools_HH

#if HAVE_SUITESPARSE_UMFPACK || defined DOXYGEN

#include <umfpack.h>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>

// primary template
template <typename T>
class UMFPackFactorizedMatrix {};

//! A class that holds an ISTL matrix factorized with UMFPack
template <int brows, int bcols>
class UMFPackFactorizedMatrix<Dune::BCRSMatrix<Dune::FieldMatrix<double, brows, bcols>>>
{
public:
    // types
    using ISTLM = Dune::BCRSMatrix<Dune::FieldMatrix<double, brows, bcols>>;
    //using IntType = int; // we use the *_di_* routines of umfpack!
    using IntType = long long; // we use the *_dl_* routines of umfpack!

    // general
    IntType n; // number of rows/columns; it is checked that the matrix is square

    // the factorization obtained from umfpack
    IntType lnz;      // number of nonzeroes in L including the diagonal
    IntType unz;      // number of nonzeroes in U
    IntType n_row;    // L is n_row -by- min(n_row,n_col)
    IntType n_col;    // U is min(n_row,n_col) -by- n_col.
    IntType nz_udiag; // matrix is singular if nz_udiag < min(n_row,n_col); A divide-by-zero will occur if nz_udiag < n_row

    IntType *Lp;
    IntType *Lj;
    double *Lx;
    IntType *Up;
    IntType *Ui;
    double *Ux;
    IntType *P;
    IntType *Q;
    IntType do_recip; //If do_recip is TRUE (one), then the scale factors Rs [i] are to be used by multiplying row i by Rs [i]. Otherwise, the entries in row i are to be divided by Rs [i].
    double *Rs;

    UMFPackFactorizedMatrix(const ISTLM &A, int verbose = 0)
    {
        // convert ISTL BCRSMatrix into UMFPacks flat compressed column format
        if (A.N() != A.M() || brows != bcols)
            throw std::invalid_argument("UMFPackFactorizedMatrix: input matrix must be square");

        n = A.N() * brows;
        IntType nnz = A.nonzeroes() * brows * bcols;
        if (verbose > 0)
            std::cout << "factorizing matrix with UMFPack n=" << n << " nnz=" << nnz << std::endl;

        // allocate space
        IntType *Ap = new (std::align_val_t{64}) IntType[n + 1]();
        IntType *Ai = new (std::align_val_t{64}) IntType[nnz]();
        double *Ax = new (std::align_val_t{64}) double[nnz]();

        // count number of entries per colum
        std::vector<IntType> column_entries(n, 0);
        for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
            for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
                for (std::size_t i = 0; i < brows; ++i)
                    for (std::size_t j = 0; j < bcols; ++j)
                        if ((*col_iter)[i][j]!=0.0)
                            column_entries[col_iter.index() * bcols + j] += 1;

        // fill the Ap array with starts of columns
        Ap[0] = 0;
        for (std::size_t j = 1; j <= n; ++j)
            Ap[j] = Ap[j - 1] + column_entries[j - 1];
        // for (size_t j=0; j<column_entries.size(); j++)
        //     std::cout << j << ": " <<  column_entries[j] << " " << Ap[j] << std::endl;
        // std::cout << "total size: " << Ap[n] << std::endl;

        // clear counters
        std::fill(column_entries.begin(), column_entries.end(), 0);

        // fill arrays
        for (auto row_iter = A.begin(); row_iter != A.end(); ++row_iter)
            for (auto col_iter = row_iter->begin(); col_iter != row_iter->end(); ++col_iter)
                for (std::size_t j = 0; j < bcols; ++j)
                {
                    std::size_t J = col_iter.index() * bcols + j;
                    for (std::size_t i = 0; i < brows; ++i)
                        if ((*col_iter)[i][j]!=0.0)
                        {
                            Ai[Ap[J] + column_entries[J]] = row_iter.index() * brows + i;
                            Ax[Ap[J] + column_entries[J]] = (*col_iter)[i][j];
                            column_entries[J] += 1;
                        }
                }
        // for (size_t j=0; j<n; j++)
        //     for (size_t k=Ap[j]; k<Ap[j+1]; k++)
        //         std::cout << Ai[k] << " " << j << " " << Ax[k] << std::endl;

        // factorize matrix
        double Control[UMFPACK_CONTROL];
        // umfpack_di_defaults(Control);
        umfpack_dl_defaults(Control);
        double Info[UMFPACK_INFO];
        Control[UMFPACK_PRL] = verbose;
        void *Symbolic; // umfpacks opaque symbolic object
        // umfpack_di_symbolic(A.N() * brows, A.M() * bcols, Ap, Ai, Ax, &Symbolic, Control, Info);
        umfpack_dl_symbolic(A.N() * brows, A.M() * bcols, Ap, Ai, Ax, &Symbolic, Control, Info);
        void *Numeric; // umfpacks opaque numerics object
        // umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
        umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
        if (verbose==1)
        {
            std::cout << "[UMFPack Factorization]" << std::endl;
            std::cout << "Wallclock Time taken: " << Info[UMFPACK_NUMERIC_WALLTIME] << " (CPU Time: " << Info[UMFPACK_NUMERIC_TIME] << ")" << std::endl;
            std::cout << "Flops taken: " << Info[UMFPACK_FLOPS] << std::endl;
            std::cout << "Peak Memory Usage: " << Info[UMFPACK_PEAK_MEMORY] * Info[UMFPACK_SIZE_OF_UNIT] << " bytes" << std::endl;
            std::cout << "Condition number estimate: " << 1. / Info[UMFPACK_RCOND] << std::endl;
            std::cout << "Numbers of non-zeroes in decomposition: L: " << Info[UMFPACK_LNZ] << " U: " << Info[UMFPACK_UNZ] << std::endl;
        }
        else if (verbose>1)
            // umfpack_di_report_info(Control, Info);
            umfpack_dl_report_info(Control, Info);

        // free space for input matrix now that we have factorized
        delete[] Ax;
        delete[] Ai;
        delete[] Ap;

        if (verbose > 1)
            std::cout << "passed factorization phase" << std::endl;

        // now externalize the factorization
        // Copies L, U, P, Q, and R from the Numeric object into arrays provided by the user. The
        // matrix L is returned in compressed row form (with the column indices in each row sorted
        // in ascending order). The matrix U is returned in compressed column form (with sorted
        // columns). There are no explicit zero entries in L and U, but such entries may exist in the
        // Numeric object. The permutations P and Q are represented as permutation vectors, where
        // P[k] = i means that row i of the original matrix is the the k-th row of PAQ, and where
        // Q[k] = j means that column j of the original matrix is the k-th column of PAQ. This is
        // identical to how MATLAB uses permutation vectors (type help colamd in MATLAB 6.1 or
        // later).

        // get sizes
        // auto status = umfpack_di_get_lunz(&lnz, &unz, &n_row, &n_col, &nz_udiag, Numeric);
        auto status = umfpack_dl_get_lunz(&lnz, &unz, &n_row, &n_col, &nz_udiag, Numeric);
        if (status != UMFPACK_OK)
            throw std::invalid_argument("UMFPackFactorizedMatrix: umfpack_dl_get_lunz returned not OK");

        if (verbose > 0)
            std::cout << "result of umfpack_d*_get_lunz" << " lnz=" << lnz << " unz=" << unz << " n_row=" << n_row << " n_col=" << n_col << " nz_udiag=" << nz_udiag << std::endl;

        if (n_row!=n || n_col!=n)
        {
            std::cout << "factorization with UMFPACK failed n=" << n << " n_row=" << n_row << " n_col=" << n_col << std::endl;
            std::cout << "result of umfpack_d*_get_lunz" << " lnz=" << lnz << " unz=" << unz << " n_row=" << n_row << " n_col=" << n_col << " nz_udiag=" << nz_udiag << std::endl;
            throw std::invalid_argument("UMFPackFactorizedMatrix: factorization failed");
        }

        if (nz_udiag < std::min(n_row,n_col))
        {
            std::cout << "nz_udiag=" << nz_udiag << " n_row=" << n_row << " n_col=" << n_col << std::endl;
            throw std::invalid_argument("UMFPackFactorizedMatrix: input matrix is singular");
        }

        if (verbose > 1)
            std::cout << "allocating memory now" << std::endl;

        // allocate space for the output arrays
        Lp = new IntType[n_row + 1]; // The n_row-by-min(n_row,n_col) matrix L is returned in compressed-row form. Each row is stored in sorted order, so diagonal is last!
        Lj = new IntType[lnz]; 
        Lx = new double[lnz];
        Up = new IntType[n_col + 1]; // The min(n_row,n_col)-by-n_col matrix U is returned in compressed-column form
        Ui = new IntType[unz];
        Ux = new double[unz];
        P = new IntType[n_row]; // permutation vector P is defined as P [k] = i, where the original row i of A is the kth pivot row in PAQ
        Q = new IntType[n_col]; // permutation vector Q is defined as Q [k] = j, where the original column j of A is the kth pivot column in PAQ
        Rs = new double[n_row]; // Row i of A is scaled by dividing or multiplying its values by Rs [i]
                                // If do_recip is TRUE (one), then the scale factors Rs [i] are to be used by multiplying row i by Rs [i]. 
                                // Otherwise, the entries in row i are to be divided by Rs [i].

        if (verbose > 1)
            std::cout << "fill data" << std::endl; 

        // status = umfpack_di_get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, nullptr, &do_recip, Rs, Numeric);
        status = umfpack_dl_get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, nullptr, &do_recip, Rs, Numeric);
        if (status != UMFPACK_OK)
            throw std::invalid_argument("UMFPackFactorizedMatrix: umfpack_dl_get_numeric returned not OK");

        // Control[UMFPACK_PRL] = 5;
        // umfpack_di_report_perm(n,P,Control);
        // umfpack_di_report_perm(n,Q,Control);

        // now we can free the opaque objects
        // umfpack_di_free_symbolic(&Symbolic);
        // umfpack_di_free_numeric(&Numeric);
        umfpack_dl_free_symbolic(&Symbolic);
        umfpack_dl_free_numeric(&Numeric);
    }

    // delete copy constructor
    UMFPackFactorizedMatrix(const UMFPackFactorizedMatrix<ISTLM> &) = delete;

    // delete assignment
    UMFPackFactorizedMatrix<ISTLM> &operator=(const UMFPackFactorizedMatrix<ISTLM> &o) = delete;

    ~UMFPackFactorizedMatrix()
    {
        // free space for factorization
        delete[] Rs;
        delete[] Q; 
        delete[] P;
        delete[] Ux;
        delete[] Ui;
        delete[] Up;
        delete[] Lx;
        delete[] Lj;
        delete[] Lp;
    }
};

//! Solve a linear system just to check if it works
template <int brows, int bcols, typename T>
void solve_linear_system (UMFPackFactorizedMatrix<Dune::BCRSMatrix<Dune::FieldMatrix<double, brows, bcols>>>& F, T& x, const T& b)
{
    std::vector<double> t1(F.n);
    std::vector<double> t2(F.n);

    // first step of preparing right hand side: row scaling and copy to x
    // input b, output t1
    if (F.do_recip)
        for (size_t i=0; i<F.n; ++i) t1[i] = b[i] * F.Rs[i];
    else
        for (size_t i=0; i<F.n; ++i) t1[i] = b[i] / F.Rs[i];

    // second step: apply row permutation P
    // input t1, output t2
    for (size_t k=0; k<F.n; ++k) 
        t2[k] = t1[F.P[k]];

    // back solve L (in compressed row storage)
    // input t2, output t1
    for (size_t i=0; i<F.n; i++)
    {
        double sum = t2[i];
        for (size_t k=F.Lp[i]; k<F.Lp[i+1]-1; k++)
            sum -= F.Lx[k]*t1[F.Lj[k]];
        t1[i] = sum;
    }

    // back solve U (in compressed column storage)
    // input t1, output t2
    for (size_t j=F.n-1; j>=0; j--)
    {
        double xj = t1[j] / F.Ux[F.Up[j+1]-1];
        for (size_t k=F.Up[j]; k<F.Up[j+1]-1; k++)
            t1[F.Ui[k]] -= F.Ux[k]*xj;
        t2[j] = xj;
    }

    // apply column permutations to solution
    // input t2, output x
    for (size_t k=0; k<F.n; k++)
        x[F.Q[k]] = t2[k];
}
#endif

#endif
