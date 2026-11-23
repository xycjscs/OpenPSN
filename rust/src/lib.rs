//! Rust sparse Cholesky (LL^T) behind the psn2d `chol_*` C ABI.
//!
//! Drop-in replacement for the C++/Eigen bridge in
//! `psn2d/memopt_cxx/eigen_chol.cpp`: same symbol names, same signatures,
//! same meaning of `status` (0 = success, 2 = `Eigen::NumericalIssue`
//! equivalent, negative = invalid input / internal error).
//!
//! Two intentional deviations from the C++ build, both documented in the
//! hand-off report:
//!   * `ordering == 0` means "factor the matrix exactly as given" (natural
//!     order).  The C++ build maps `ordering == 0` to AMD.  The solver only
//!     ever passes `ordering == 1`.
//!   * `ordering != 0` requires METIS (found via `RTLD_DEFAULT`, then
//!     `$OPENPSN_METIS_LIB`, then the usual SONAMEs).  If METIS cannot be
//!     resolved the call FAILS (status -3) rather than silently falling back
//!     to natural order and a 10x fill blow-up.
//!
//! Extra entry points (the original four are unchanged):
//!   * `chol_solve2(h, n, b0, x0, b1, x1)` — two right-hand sides through the
//!     factor in ONE pass (`SharedChol` always does exactly two solves on the
//!     same factor back to back); bit-for-bit the same result as two
//!     `chol_solve` calls.
//!   * `chol_solve_k(h, n, const double* const* b, double* const* x, int k)`.
//!   * `chol_last_error()` -> `const char*`, and `chol_diag(h, out, nout)`.
//!
//! `chol_solve*` never writes through the handle: the substitution work
//! buffer is thread-local, so forked sweep workers keep sharing the factor
//! copy-on-write.

use std::cell::RefCell;
use std::os::raw::{c_char, c_double, c_int, c_longlong, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

mod chol;
mod metis;
mod packed;
mod panels;
mod sp;

/// One factorization handle.  Immutable after construction.
pub struct Handle {
    n: usize,
    nnz_l: i64,
    /// internal row -> original row (composite fill-reducing order + postorder)
    map: Vec<i32>,
    pl: panels::Panels,
}

thread_local! {
    static LAST_ERR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

fn set_status(status: *mut c_int, v: i32) {
    if !status.is_null() {
        unsafe { *status = v };
    }
}

fn set_err(msg: &str) {
    LAST_ERR.with(|c| {
        let mut cell = c.borrow_mut();
        cell.clear();
        cell.extend_from_slice(msg.as_bytes());
        cell.push(0);
    });
}

/// Column-block width of the packed panels (compile-time cap `SMAX`).
fn sblock() -> usize {
    std::env::var("PSN_CHOL_S")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, panels::SMAX)
}

impl Handle {
    #[inline]
    unsafe fn solve(&self, bs: &[*const f64], xs: &[*mut f64]) {
        let k = bs.len();
        let n = self.n;
        SCRATCH.with(|c| {
            let mut cell = c.borrow_mut();
            let need = n * k;
            if cell.len() < need {
                cell.resize(need, 0.0);
            }
            let t = cell.as_mut_ptr();
            let map = self.map.as_ptr();
            // b -> t  in the factor's own ordering: writing every entry means
            // the scratch never has to be cleared, for any number of RHS.
            for q in 0..k {
                let bq = *bs.get_unchecked(q);
                let tq = t.add(q * n);
                for i in 0..n {
                    *tq.add(i) = *bq.add(*map.add(i) as usize);
                }
            }
            self.pl.solve_generic(n, t, k);
            for q in 0..k {
                let xq = *xs.get_unchecked(q);
                let tq = t.add(q * n);
                for i in 0..n {
                    *xq.add(*map.add(i) as usize) = *tq.add(i);
                }
            }
        });
    }
}

unsafe fn nan_fill(x: *mut f64, n: usize) {
    // fail loud: a NaN solution propagates into every downstream residual check
    ptr::write_bytes(x as *mut u8, 0xff, n * 8);
}

// --------------------------------------------------------------------------- #
// C ABI
// --------------------------------------------------------------------------- #

#[no_mangle]
pub unsafe extern "C" fn chol_factor(
    n: c_int,
    nz: c_int,
    cp: *const c_int,
    ci: *const c_int,
    cx: *const c_double,
    ordering: c_int,
    status: *mut c_int,
) -> *mut c_void {
    set_status(status, 0);
    set_err("");
    if cp.is_null() || ci.is_null() || cx.is_null() {
        set_status(status, -2);
        set_err("null CSC array");
        return ptr::null_mut();
    }
    if n <= 0 || nz < 0 {
        set_status(status, -2);
        set_err("non-positive dimension or nnz");
        return ptr::null_mut();
    }
    let n = n as usize;
    let nnz = nz as usize;
    let ap = std::slice::from_raw_parts(cp, n + 1);
    let ai = std::slice::from_raw_parts(ci, nnz);
    let ax = std::slice::from_raw_parts(cx, nnz);
    let sblk = sblock();
    let built = catch_unwind(AssertUnwindSafe(|| -> Result<Handle, (String, i32)> {
        sp::check_csc(n, nnz, ap, ai, ax).map_err(|m| (m, -2))?;
        let (l, map) = chol::analyze_and_factor(n, ap, ai, ax, ordering)?;
        let nnz_l = l.lp[n] as i64;
        let pl = panels::build(&l, sblk).map_err(|m| (m, -4))?;
        drop(l);
        Ok(Handle { n, nnz_l, map, pl })
    }));
    match built {
        Ok(Ok(h)) => Box::into_raw(Box::new(h)) as *mut c_void,
        Ok(Err((msg, code))) => {
            set_status(status, code);
            set_err(&msg);
            ptr::null_mut()
        }
        Err(_) => {
            set_status(status, -1);
            set_err("panic inside chol_factor");
            ptr::null_mut()
        }
    }
}

impl Handle {
    /// Bytes held by the factor (diagnostics only).
    #[allow(dead_code)]
    fn bytes(&self) -> usize {
        self.pl.bytes() + self.map.len() * 4
    }
}

/// `x = A^-1 b` (single RHS).  Read-only with respect to `h`.
#[no_mangle]
pub unsafe extern "C" fn chol_solve(
    h: *mut c_void,
    n: c_int,
    b: *const c_double,
    x: *mut c_double,
) {
    if x.is_null() {
        return;
    }
    if h.is_null() || n <= 0 {
        return;
    }
    let hdl = &*(h as *const Handle);
    if hdl.n != n as usize {
        nan_fill(x, n as usize);
        set_err("chol_solve: n does not match the factorization");
        return;
    }
    if b.is_null() {
        nan_fill(x, n as usize);
        return;
    }
    let r = catch_unwind(AssertUnwindSafe(|| hdl.solve(&[b], &[x])));
    if r.is_err() {
        nan_fill(x, n as usize);
        set_err("panic inside chol_solve");
    }
}

/// Two right-hand sides through the factor in ONE pass.  Identical results to
/// two `chol_solve` calls; roughly half the factor traffic.
#[no_mangle]
pub unsafe extern "C" fn chol_solve2(
    h: *mut c_void,
    n: c_int,
    b0: *const c_double,
    x0: *mut c_double,
    b1: *const c_double,
    x1: *mut c_double,
) {
    if x0.is_null() || x1.is_null() || h.is_null() || n <= 0 {
        return;
    }
    let hdl = &*(h as *const Handle);
    if hdl.n != n as usize || b0.is_null() || b1.is_null() {
        nan_fill(x0, n as usize);
        nan_fill(x1, n as usize);
        return;
    }
    let r = catch_unwind(AssertUnwindSafe(|| hdl.solve(&[b0, b1], &[x0, x1])));
    if r.is_err() {
        nan_fill(x0, n as usize);
        nan_fill(x1, n as usize);
        set_err("panic inside chol_solve2");
    }
}

/// `k` right-hand sides; `b`/`x` are arrays of `k` pointers to length-`n`
/// blocks.  k = 1, 2, 3, 4 run fused in a single pass, larger k loop.
#[no_mangle]
pub unsafe extern "C" fn chol_solve_k(
    h: *mut c_void,
    n: c_int,
    b: *const *const c_double,
    x: *mut *mut c_double,
    k: c_int,
) {
    if h.is_null() || n <= 0 || k <= 0 || b.is_null() || x.is_null() {
        return;
    }
    let hdl = &*(h as *const Handle);
    let k = k as usize;
    let bs = std::slice::from_raw_parts(b, k);
    let xs = std::slice::from_raw_parts_mut(x, k);
    if hdl.n != n as usize || bs.iter().any(|p| p.is_null()) || xs.iter().any(|p| p.is_null()) {
        for q in 0..k {
            nan_fill(*xs.get_unchecked(q), n as usize);
        }
        return;
    }
    let r = catch_unwind(AssertUnwindSafe(|| hdl.solve(bs, xs)));
    if r.is_err() {
        for q in 0..k {
            nan_fill(*xs.get_unchecked(q), n as usize);
        }
        set_err("panic inside chol_solve_k");
    }
}

/// Number of nonzeros in the Cholesky factor L (comparable to the C++
/// bridge's `chol_nnz`).
#[no_mangle]
pub unsafe extern "C" fn chol_nnz(h: *mut c_void) -> c_longlong {
    if h.is_null() {
        return -1;
    }
    let hdl = &*(h as *const Handle);
    hdl.nnz_l as c_longlong
}

/// Diagnostics: out = [nnz_L, npanels, row_index_entries, stored_values,
/// bytes_held, n].  Returns the number of entries written.
#[no_mangle]
pub unsafe extern "C" fn chol_diag(h: *mut c_void, out: *mut c_longlong, nout: c_int) -> c_int {
    if h.is_null() || out.is_null() || nout <= 0 {
        return 0;
    }
    let hdl = &*(h as *const Handle);
    let vals = [
        hdl.nnz_l,
        hdl.pl.npanels() as i64,
        hdl.pl.panel_rows() as i64,
        hdl.pl.stored_vals() as i64,
        (hdl.pl.bytes() + hdl.map.len() * 4) as i64,
        hdl.n as i64,
    ];
    let m = (nout as usize).min(vals.len());
    std::ptr::copy_nonoverlapping(vals.as_ptr(), out, m);
    m as c_int
}

/// Message for the last failed call on this thread ("" when there was none).
/// The pointer is valid until the next failing call on the same thread.
#[no_mangle]
pub unsafe extern "C" fn chol_last_error() -> *const c_char {
    LAST_ERR.with(|c| c.borrow().as_ptr() as *const c_char)
}

#[no_mangle]
pub unsafe extern "C" fn chol_free(h: *mut c_void) {
    if !h.is_null() {
        drop(Box::from_raw(h as *mut Handle));
    }
}

// --------------------------------------------------------------------------- #
// B.3 paraboloidal source update (the outer loop's hottest non-transport call)
// --------------------------------------------------------------------------- #

/// Groups the accumulator block is sized for; larger `ng` runs in chunks.
const SRC_GMAX: usize = 16;

#[inline(always)]
unsafe fn src_update(
    ng: usize,
    nodes: usize,
    base: *const f64,
    nu: *const f64,
    chi: *const f64,
    lam: f64,
    qim: *const f64,
    out: *mut f64,
) {
    for g0 in (0..ng).step_by(SRC_GMAX) {
        let gb = SRC_GMAX.min(ng - g0);
        for n in 0..nodes {
            // acc[g][k] for this node only: 16 x 5 doubles, L1-resident
            let mut acc = [[0f64; 5]; SRC_GMAX];
            for gg in 0..ng {
                let v = *nu.add(n * ng + gg);
                let qp = qim.add((gg * nodes + n) * 5);
                let q = &*(qp as *const [f64; 5]);
                for g in 0..gb {
                    // left to right, as `Sgg + chi[g] * nuSf[g2] / lam` reads;
                    // no FMA -- LLVM does not contract without fast-math
                    let c = *base.add((n * ng + g0 + g) * ng + gg)
                        + (*chi.add(g0 + g) * v) / lam;
                    let a = acc.get_unchecked_mut(g);
                    a[0] += c * q[0];
                    a[1] += c * q[1];
                    a[2] += c * q[2];
                    a[3] += c * q[3];
                    a[4] += c * q[4];
                }
            }
            for g in 0..gb {
                let op = out.add(((g0 + g) * nodes + n) * 5) as *mut [f64; 5];
                *op = acc[g];
            }
        }
    }
}

/// `out[g,n,k] = sum_G (base[n,g,G] + chi[g]*nu[n,G]/lam) * qim[G,n,k]`
///
/// The NumPy version this replaces materialises the `(ng, ng, nodes)`
/// coefficient array and then reduces over the input group with a stride of
/// `nodes` -- seven streams through 12 MiB to produce 2.8 MiB.  Here the
/// coefficient array is never built, and the caller passes the two scatter
/// tables **node-major** (`base` `(nodes, ng, ng)`, `nu` `(nodes, ng)`), which
/// costs one pass per run because the node -> material map never changes, so
/// one node's whole 7 x 7 (plus its 7 fission weights) is a single contiguous
/// read.  Each array is touched exactly once and `out` is written once instead
/// of fifty temporaries.
///
/// Bit-for-bit the same arithmetic as the loop it replaces: the sum over `G`
/// runs ascending from exact zero, `chi*nu` and the division by `lam` stay
/// separate operations, and no FMA is used.  `dev/srcident.py` checks the bytes
/// against the Python loop; a single flipped ulp would also show up as a
/// different sweep count in the three gate cases.
#[no_mangle]
pub unsafe extern "C" fn psn_src_update(
    ng: c_int,
    nodes: c_int,
    base: *const c_double,
    nu: *const c_double,
    chi: *const c_double,
    lam: c_double,
    qim: *const c_double,
    out: *mut c_double,
    status: *mut c_int,
) {
    set_status(status, 0);
    set_err("");
    if ng <= 0 || nodes <= 0 || base.is_null() || nu.is_null() || chi.is_null()
        || qim.is_null() || out.is_null()
    {
        set_status(status, -2);
        set_err("psn_src_update: null array or empty shape");
        if !out.is_null() && ng > 0 && nodes > 0 {
            nan_fill(out, (ng * nodes * 5) as usize);
        }
        return;
    }
    let (ng, nodes) = (ng as usize, nodes as usize);
    let r = catch_unwind(AssertUnwindSafe(|| {
        src_update(ng, nodes, base, nu, chi, lam, qim, out)
    }));
    if r.is_err() {
        nan_fill(out, ng * nodes * 5);
        set_status(status, -1);
        set_err("panic inside psn_src_update");
    }
}
