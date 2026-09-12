// Portable C++ bridge: exact sparse Cholesky (Eigen SimplicialLLT) with a
// plain C ABI so Python can share the factor COW across forked sweep
// workers without any Python-side factor copy.
//
// Matrix input: column-major CSC (Eigen native): indptr (n+1), indices (nz),
// values (nz).  ordering: 0 = AMD, 1 = METIS (needs libmetis).
//
// The factor is a C++ heap object (Holder).  chol_solve / sparse_mrsolve
// are const: they never write the factor, so forked children sharing it
// via COW are safe.
//
// sparse_mrsolve: multi-RHS solve of A x = B where B is SPARSE (few nnz
// per column).  Each solution column's support is confined to the fill
// subtree of its RHS support (worklist substitution), so the cost is
// O(|subtree|) per column, not O(nnz(L)).  Parallel over RHS columns
// (independent).  See the C-ABI comment for the array layouts.
//
// Build:  python build.py --eigen-include DIR [--metis-include DIR --metis-lib DIR]
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>   // required by Eigen MetisSupport — must precede it
#include <memory>
#include <mutex>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#ifdef OPENPSN_WITH_METIS
#include <Eigen/MetisSupport>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using Sp = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Vec = Eigen::VectorXd;
struct Holder {
  virtual ~Holder() {}
  virtual int compute(const Sp&) = 0;
  virtual void solve(const double*, double*, int) const = 0;
  virtual long long nnz() const = 0;
  // raw views into the factor's Cholesky factor L (col-major CSC, ascending
  // within column, diagonal = FIRST entry of each column) and permutations
  virtual const int* l_p() const = 0;
  virtual const int* l_i() const = 0;
  virtual const double* l_v() const = 0;
  virtual const int* perm() const = 0;    // P  : (P b)[i] = b[P[i]]
  virtual const int* pinv() const = 0;    // Pinv
  // sparse multi-RHS solve; see the C-ABI doc at the bottom
  virtual void sparse_mrsolve(const int* lrp, const int* lri, int k,
                              const int* b_indptr, const int* b_idx,
                              const double* b_val,
                              int* x_indptr, int* x_idx,
                              double* x_val) const = 0;
};

// ---- tiny binary heaps (min for forward, max for backward) ------------- //
struct MinHeap {
  std::vector<int> v;
  void push(int x) {
    v.push_back(x);
    int c = v.size() - 1;
    while (c > 0) {
      int p = (c - 1) / 2;
      if (v[p] <= v[c]) break;
      std::swap(v[p], v[c]);
      c = p;
    }
  }
  int pop() {
    int r = v[0];
    v[0] = v.back();
    v.pop_back();
    int c = 0;
    for (;;) {
      int l = 2 * c + 1, r2 = 2 * c + 2, m = c;
      if (l < (int)v.size() && v[l] < v[m]) m = l;
      if (r2 < (int)v.size() && v[r2] < v[m]) m = r2;
      if (m == c) break;
      std::swap(v[m], v[c]);
      c = m;
    }
    return r;
  }
  bool empty() const { return v.empty(); }
};
struct MaxHeap {
  std::vector<int> v;
  void push(int x) {
    v.push_back(x);
    int c = v.size() - 1;
    while (c > 0) {
      int p = (c - 1) / 2;
      if (v[p] >= v[c]) break;
      std::swap(v[p], v[c]);
      c = p;
    }
  }
  int pop() {
    int r = v[0];
    v[0] = v.back();
    v.pop_back();
    int c = 0;
    for (;;) {
      int l = 2 * c + 1, r2 = 2 * c + 2, m = c;
      if (l < (int)v.size() && v[l] > v[m]) m = l;
      if (r2 < (int)v.size() && v[r2] > v[m]) m = r2;
      if (m == c) break;
      std::swap(v[m], v[c]);
      c = m;
    }
    return r;
  }
  bool empty() const { return v.empty(); }
};

template <class Order> struct Factor : Holder {
  Eigen::SimplicialLLT<Sp, Eigen::Lower, Order> f;
  int compute(const Sp& a) override { f.compute(a); return int(f.info()); }
  void solve(const double* b, double* x, int n) const override {
    Eigen::Map<Vec>(x, n) = f.solve(Eigen::Map<const Vec>(b, n));
  }
  long long nnz() const override {
    // cp[n] is authoritative — nonZeros() may undercount (initial capacity)
    return f.matrixL().nestedExpression().outerIndexPtr()[int(f.cols())];
  }
  const int* l_p() const override {
    return f.matrixL().nestedExpression().outerIndexPtr();
  }
  const int* l_i() const override {
    return f.matrixL().nestedExpression().innerIndexPtr();
  }
  const double* l_v() const override {
    return f.matrixL().nestedExpression().valuePtr();
  }
  const int* perm() const override {
    return f.permutationP().indices().data();
  }
  const int* pinv() const override {
    return f.permutationPinv().indices().data();
  }
  void sparse_mrsolve(const int* lrp, const int* lri, int k,
                      const int* b_indptr, const int* b_idx,
                      const double* b_val,
                      int* x_indptr, int* x_idx, double* x_val)
      const override {
    const int n = int(f.cols());
    const int* lp = l_p();
    const int* li = l_i();
    const double* lv = l_v();
    const int* P = perm();
    const int* Pinv = pinv();

    // per-thread scratch (z: permuted-space values, x: permuted-space result).
    // CRITICAL: sizing happens INSIDE the parallel region, below — each OMP
    // worker owns its own thread_local copy, and only the calling thread
    // runs code before the pragma. Sizing here (master only) left workers'
    // vectors empty (nullptr) -> segfault when a worker grabs an iteration
    // (schedule(dynamic,128) + k<128 makes exactly one thread do all work,
    // and that thread is a worker ~half the time at T>=16).
    thread_local std::vector<double> zbuf, xbuf;
    thread_local std::vector<unsigned char> zflag, xflag, zpend, xpend;

    // per-RHS-col output, r-owned (no cross-thread races); packed serially
    std::vector<std::vector<std::pair<int, double>>> xo(k);
    #pragma omp parallel for schedule(dynamic, 128)
    for (int r = 0; r < k; r++) {
      if ((int)zbuf.size() < n) {   // each thread sizes its OWN copy; no race
        zbuf.assign(n, 0.0);
        xbuf.assign(n, 0.0);
        zflag.assign(n, 0);
        xflag.assign(n, 0);
        zpend.assign(n, 0);
        xpend.assign(n, 0);
      }
      std::vector<int> ztouched, xtouched;
      MinHeap hf;
      MaxHeap hb;

      // 1) w = Pinv b:  w[i] = b[Pinv[i]]  =>  i = P[idx]
      //    (stored factor satisfies L L^T = Pinv A Pinv^T — verified
      //    element-wise against chol_solve; do NOT swap with P here)
      for (int a = b_indptr[r]; a < b_indptr[r + 1]; a++) {
        int i = P ? P[b_idx[a]] : b_idx[a];
        if (!zflag[i]) { zflag[i] = 1; ztouched.push_back(i); }
        zbuf[i] += b_val[a];
      }
      // 2) forward substitution z = L^-1 y, worklist over the fill subtree
      for (int i : ztouched) {
        if (zbuf[i] != 0.0) { zpend[i] = 1; hf.push(i); }
      }
      while (!hf.empty()) {
        int i = hf.pop();
        zpend[i] = 0;
        int s = lp[i];
        double zi = zbuf[i] / lv[s];   // divide by the diagonal L[i,i]
        if (zi == 0.0) continue;
        zbuf[i] = zi;                  // store: backward pass reads z, not y
        int e = lp[i + 1];
        for (int t = s + 1; t < e; t++) {   // off-diagonal: (row li[t], val lv[t])
          int rr = li[t];
          double old = zbuf[rr];
          double nv = old - lv[t] * zi;
          zbuf[rr] = nv;
          if (old == 0.0) {
            if (!zflag[rr]) { zflag[rr] = 1; ztouched.push_back(rr); }
            if (nv != 0.0 && !zpend[rr]) { zpend[rr] = 1; hf.push(rr); }
          }
        }
      }
      // 3) backward substitution x = L^-T z, max-heap, push parents via the
      //    row view of L (lrp/lri)
      for (int i : ztouched) {
        xpend[i] = 1;
        hb.push(i);
      }
      while (!hb.empty()) {
        int rr = hb.pop();
        xpend[rr] = 0;
        int s = lp[rr], e = lp[rr + 1];
        double sum = 0.0;
        for (int t = s + 1; t < e; t++) {   // L[srow, rr] with srow > rr
          sum += lv[t] * xbuf[li[t]];
        }
        double xr = (zbuf[rr] - sum) / lv[s];  // diagonal = first entry
        if (xr != 0.0) {
          xbuf[rr] = xr;
          if (!xflag[rr]) { xflag[rr] = 1; xtouched.push_back(rr); }
          // parents of rr = { j < rr : L[rr, j] != 0 } = row rr of L,
          // skip the diagonal (j == rr) or we re-push rr forever
          for (int t = lrp[rr]; t < lrp[rr + 1]; t++) {
            int j = lri[t];
            if (j >= rr) continue;
            if (!xpend[j]) { xpend[j] = 1; hb.push(j); }
          }
        }
      }
      // 4) unpermute: x[k] = u[P[k]]  =>  value u[i] lands at k = Pinv[i].
      //    (P/Pinv are the INVERSE of each other; do not use P here —
      //     verified element-wise against chol_solve, see viewdiag notes.)
      auto& out = xo[r];
      out.clear();
      out.reserve(xtouched.size());
      for (int i : xtouched)
        if (xbuf[i] != 0.0)
          out.emplace_back(Pinv ? Pinv[i] : i, xbuf[i]);
      // 5) cleanup
      for (int i : xtouched) { xbuf[i] = 0.0; xflag[i] = 0; }
      for (int i : ztouched) { zbuf[i] = 0.0; zflag[i] = 0; }
    }
    // serial pack (no races)
    long long s = 0;
    for (int r = 0; r < k; r++) {
      x_indptr[r] = int(s);
      for (const auto& pr : xo[r]) {
        x_idx[s] = pr.first;
        x_val[s] = pr.second;
        s++;
      }
    }
    x_indptr[k] = int(s);
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
// raw factor views (pointers valid until chol_free)
const int* chol_l_p(void* h) { return static_cast<Holder*>(h)->l_p(); }
const int* chol_l_i(void* h) { return static_cast<Holder*>(h)->l_i(); }
const double* chol_l_v(void* h) { return static_cast<Holder*>(h)->l_v(); }
const int* chol_perm(void* h) { return static_cast<Holder*>(h)->perm(); }
const int* chol_pinv(void* h) { return static_cast<Holder*>(h)->pinv(); }
/* sparse_mrsolve(h, lrp, lri, k, b_indptr, b_idx, b_val,
                 x_indptr, x_idx, x_val)
   A x = B,  A SPD, factor A(P,P) = L L^T held at h.
   lrp/lri: CSR row view of the SAME L (row i: columns j<=i ascending) —
            built once in Python from the chol_l_* views (L.T.tocsr()).
   k: number of RHS.  B column r: b_idx[b_indptr[r]..b_indptr[r+1]] hold the
   (original-space) row indices, b_val the values.
   Output column r: x_idx[x_indptr[r]..x_indptr[r+1]] / x_val — UNSORTED
   (coo-style), caller sums duplicates.
   Only const on the factor; thread-safe; OpenMP over r.
*/
void chol_sparse_mrsolve(void* h, const int* lrp, const int* lri, int k,
                         const int* b_indptr, const int* b_idx,
                         const double* b_val,
                         int* x_indptr, int* x_idx, double* x_val) {
  static_cast<Holder*>(h)->sparse_mrsolve(lrp, lri, k, b_indptr, b_idx,
                                          b_val, x_indptr, x_idx, x_val);
}
void chol_free(void* h) { delete static_cast<Holder*>(h); }
}  // extern "C"
