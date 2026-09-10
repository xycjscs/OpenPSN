// Portable C++ bridge: exact sparse Cholesky (Eigen SimplicialLLT) with a
// plain C ABI so Python can share the factor COW across forked sweep
// workers without any Python-side factor copy.
//
// Matrix input: column-major CSC (Eigen native): indptr (n+1), indices (nz),
// values (nz).  ordering: 0 = AMD, 1 = METIS (needs libmetis).
//
// The factor is a C++ heap object (Holder).  chol_solve is const: it never
// writes the factor, so forked children sharing it via COW are safe.
//
// Build:  python build.py --eigen-include DIR [--metis-include DIR --metis-lib DIR]
#include <cstdint>
#include <exception>
#include <iostream>   // required by Eigen MetisSupport — must precede it
#include <memory>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#ifdef OPENPSN_WITH_METIS
#include <Eigen/MetisSupport>
#endif

using Sp = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Vec = Eigen::VectorXd;
struct Holder {
  virtual ~Holder() {}
  virtual int compute(const Sp&) = 0;
  virtual void solve(const double*, double*, int) const = 0;
  virtual long long nnz() const = 0;
};
template <class Order> struct Factor : Holder {
  Eigen::SimplicialLLT<Sp, Eigen::Lower, Order> f;
  int compute(const Sp& a) override { f.compute(a); return int(f.info()); }
  void solve(const double* b, double* x, int n) const override {
    Eigen::Map<Vec>(x, n) = f.solve(Eigen::Map<const Vec>(b, n));
  }
  long long nnz() const override {
    return f.matrixL().nestedExpression().nonZeros();
  }
};
extern "C" {
void* chol_factor(int n, int nz, const int* p, const int* i, const double* x,
                  int ordering, int* status) {
  try {
    Eigen::Map<const Sp> mapped(n, n, nz, p, i, x);
    Sp a(mapped);
    std::unique_ptr<Holder> h;
#ifdef OPENPSN_WITH_METIS
    if (ordering == 1)
      h.reset(new Factor<Eigen::MetisOrdering<int>>());
    else
      h.reset(new Factor<Eigen::AMDOrdering<int>>());
#else
    h.reset(new Factor<Eigen::AMDOrdering<int>>());
#endif
    *status = h->compute(a);
    if (*status) { return nullptr; }
    return h.release();
  } catch (...) { *status = -1; return nullptr; }
}
void chol_solve(void* h, int n, const double* b, double* x) {
  static_cast<Holder*>(h)->solve(b, x, n);
}
long long chol_nnz(void* h) { return static_cast<Holder*>(h)->nnz(); }
void chol_free(void* h) { delete static_cast<Holder*>(h); }
}  // extern "C"
