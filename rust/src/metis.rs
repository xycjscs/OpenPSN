// METIS nested dissection, reached through dlopen/dlsym (no build-time
// dependency, no C code): the Python side already dlopens the library with
// RTLD_GLOBAL, so `RTLD_DEFAULT` usually finds METIS_NodeND directly; the
// OPENPSN_METIS_LIB path and the usual SONAMEs are the fallbacks.
use std::env;
use std::ffi::CString;
use std::os::raw::{c_int, c_void};

use crate::sp;

// The METIS on this box (conda-forge metis 5.1.0, pkg hD0bcaf9_1007 — the one
// `dev/env.sh` points at via OPENPSN_METIS_LIB, and the header shipped next to
// it) does NOT use the stock 6-argument prototype.  metis.h:196 declares
//
//   int METIS_NodeND(idx_t *nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt,
//                    idx_t *options, idx_t *perm, idx_t *iperm);
//
// i.e. an extra trailing `iperm`, IDXTYPEWIDTH 32, and it returns 1 (not 0) on
// success.  Calling it with the stock 6 arguments makes it write `iperm` through
// a garbage 7th argument: heap corruption / SIGSEGV, which is exactly what the
// first attempts here did.
type NodeNdFn = unsafe extern "C" fn(
    *mut c_int,
    *const c_int,
    *const c_int,
    *const c_int,
    *const c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;

const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100;
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut(); // lookup in the global scope

/// METIS 5.1.0 keeps its control block and random state in process globals, so
/// concurrent `METIS_NodeND` calls race on them.  The prefactor runs 6 factorisations
/// at once (`psn._prefactor_workers`), and measured with a pinned seed that meant
/// 6 different orderings for 6 *identical* matrices — per-system `factor_nnz` of
/// 4.378M..4.451M against a single-threaded 4,374,992, and a pool total that moved
/// between processes (92.60M / 92.72M) instead of sitting at the deterministic
/// 91.87M.  So the race cost reproducibility, ~0.8% of the factor pool, and the
/// word "reproducible"; the ordering is only ever accidentally valid because
/// `node_nd`'s perm/iperm check happens to catch a torn result.
///
/// One process-wide lock serialises the ordering calls; the LL^T itself stays
/// fully parallel, so what this costs is only the ND share of a factorisation
/// (`dev/prefactorwall.py` measures the prefactor wall with and without it).
static METIS_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

fn metis_guard() -> std::sync::MutexGuard<'static, ()> {
    // A poisoned lock means some earlier factorisation panicked; carrying on is
    // strictly better than turning one bad system into a panic in every worker.
    METIS_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Ordering cache, keyed by a hash of the graph's own bytes.
///
/// Serialising METIS exposed how much duplicate work the prefactor does: on
/// `M16_S2` the 21 systems are assembled from ONE sparsity pattern (verified —
/// the CSC index arrays of (0,0), (1,3) and (2,5) hash identically), so the
/// nested dissection was being recomputed 21 times for the same graph, and ND
/// is roughly half of a Rust factorisation (per-system median 3.85 s with 6
/// racing calls, 7.42 s with them serialised).  Cached, the whole prefactor
/// dissects once, which is what pays for the lock.
///
/// Reusing an ordering across two matrices with an identical pattern is exactly
/// correct — an ordering is a function of the pattern, and any permutation is a
/// valid one; only fill and rounding depend on which.  The key is the graph
/// itself (n plus both index arrays), so a different pattern can never hit.  The
/// hash costs ~1 ms against the 9 MB of indices, next to seconds of dissection.
static ORDER_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(u64, usize), Vec<i32>>>,
> = std::sync::OnceLock::new();

/// Cache key: the graph AND the option vector, because the ordering is a
/// function of both.  Options used to be left out, which made every
/// `PSN_METIS_SEED/NCUTS/NITER` after the first call silently return the first
/// call's ordering — harmless while the env is constant for a whole process
/// (the normal case), and exactly what dev/fillprobe.py was designed to vary.
fn graph_key(n: usize, xadj: &[i32], adj: &[i32], opts: Option<&[c_int]>) -> (u64, usize) {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    n.hash(&mut h);
    xadj.hash(&mut h);
    adj.hash(&mut h);
    match opts {
        None => (-1i32).hash(&mut h), // NULL options are a distinct config
        Some(o) => o.hash(&mut h),
    }
    (h.finish(), n)
}

extern "C" {
    fn dlopen(filename: *const i8, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
}

unsafe fn lookup(handle: *mut c_void) -> *mut c_void {
    dlsym(handle, b"METIS_NodeND\0".as_ptr() as *const i8)
}

/// Resolve `METIS_NodeND`; `None` plus a human-readable reason on failure.
unsafe fn resolve() -> Result<NodeNdFn, String> {
    let mut tried: Vec<String> = Vec::new();
    let p = lookup(RTLD_DEFAULT);
    if !p.is_null() {
        return Ok(std::mem::transmute::<*mut c_void, NodeNdFn>(p));
    }
    tried.push("RTLD_DEFAULT".into());
    let mut cands: Vec<String> = Vec::new();
    if let Ok(v) = env::var("OPENPSN_METIS_LIB") {
        cands.push(v);
    }
    for s in ["libmetis.so", "libmetis.so.5", "libmetis.so.4"] {
        cands.push(s.into());
    }
    for c in cands {
        let cs = match CString::new(c.clone()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let h = dlopen(cs.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
        if h.is_null() {
            tried.push(format!("{c} (dlopen failed)"));
            continue;
        }
        let p = lookup(h);
        if p.is_null() {
            tried.push(format!("{c} (no METIS_NodeND)"));
            continue;
        }
        return Ok(std::mem::transmute::<*mut c_void, NodeNdFn>(p));
    }
    Err(format!("METIS_NodeND not found; tried {}", tried.join(", ")))
}

/// METIS 5 option indices (`moptions_et` in metis.h — this is the 5.x order,
/// not 4.x's) and the array length.
const OPT_NITER: usize = 6;
const OPT_NCUTS: usize = 7;
const OPT_SEED: usize = 8;
const NOPTIONS: usize = 40;
/// `METIS_DEFAULT`: "use your own default", i.e. what NULL options meant.
const DEFAULT_OPT: c_int = -1;
/// Pinned seed.  The value is arbitrary and deliberately not 0.
const DEFAULT_SEED: c_int = 1;

fn opt_from_env(name: &str) -> Option<c_int> {
    env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<c_int>().ok())
}

/// Options for `METIS_NodeND`.
///
/// The kernel used to pass NULL here, which is *not* "the deterministic
/// default": METIS 5.1.0's default `SEED` is -1 and its once-per-process
/// `InitCtrl` turns that into a clock-derived seed.  So every process
/// factorised a different permutation of the same matrix — measured on
/// `M16_S2`, three identical runs gave `factor_nnz` totals of 93,115,367 /
/// 93,067,131 / 92,957,944 (0.17% spread).  That single fact is the source of
/// the solver's run-to-run 1-ulp floor, of the ±0.17% random size of the
/// factor pool, and of the Dombey tail's 451-578-sweep wobble; the word
/// "reproducible" was unavailable because of it.  Passing an explicit array
/// with a fixed seed makes the ordering, the fill and the arithmetic repeat
/// run to run.
///
/// `PSN_METIS_SEED` re-pins to another seed, `PSN_METIS_NCUTS` /
/// `PSN_METIS_NITER` ask for more refinement (lower fill, bigger cost), and
/// `PSN_METIS_NULLOPT=1` restores the old NULL call for an A/B without a
/// rebuild.
fn options() -> Option<Vec<c_int>> {
    if env::var("PSN_METIS_NULLOPT").map(|v| v == "1").unwrap_or(false) {
        return None;
    }
    let mut o = vec![DEFAULT_OPT; NOPTIONS];
    o[OPT_SEED] = opt_from_env("PSN_METIS_SEED").unwrap_or(DEFAULT_SEED);
    if let Some(v) = opt_from_env("PSN_METIS_NCUTS") {
        o[OPT_NCUTS] = v;
    }
    if let Some(v) = opt_from_env("PSN_METIS_NITER") {
        o[OPT_NITER] = v;
    }
    Some(o)
}

/// Nested-dissection ordering: returns `q` with `q[position] = vertex` (the
/// elimination order — the top-level separator comes last, verified on path and
/// 2-D-grid graphs against this exact library).
///
/// Errors (rather than falling back to natural order) so the caller fails
/// loudly instead of silently producing a 10x fill blow-up.
pub fn node_nd(n: usize, xadj: &[i32], adj: &[i32]) -> Result<Vec<i32>, String> {
    if n < 2 {
        return Ok((0..n).map(|v| v as i32).collect());
    }
    let opts = options();
    let key = graph_key(n, xadj, adj, opts.as_deref());
    if let Some(hit) = ORDER_CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .cloned()
    {
        // Cached entries were validated when they were computed, and a Vec<i32>
        // copied out of the cache cannot have been torn into.
        return Ok(hit);
    }
    let f = unsafe { resolve()? };
    let mut perm = vec![0i32; n]; // position -> vertex
    let mut iperm = vec![0i32; n]; // vertex   -> position
    let mut nvtxs = n as c_int;
    let nulp = opts
        .as_ref()
        .map(|o| o.as_ptr())
        .unwrap_or(std::ptr::null());
    // PSN_METIS_NOLOCK=1 reproduces the racy concurrent call, for the A/B only.
    let nolock = env::var("PSN_METIS_NOLOCK")
        .map(|v| v == "1")
        .unwrap_or(false);
    let _guard = if nolock {
        None
    } else {
        Some(metis_guard())
    };
    let rc = unsafe {
        f(
            &mut nvtxs,
            xadj.as_ptr(),
            adj.as_ptr(),
            std::ptr::null(),
            nulp,
            perm.as_mut_ptr(),
            iperm.as_mut_ptr(),
        )
    };
    drop(_guard);
    // This build signals success with 1; the arrays are the real gate.
    if rc < 0 {
        return Err(format!("METIS_NodeND returned rc={}", rc));
    }
    // perm and iperm must be exact inverses of each other: this is what proves
    // the 7-argument prototype was matched (a wrong arity corrupts the heap
    // instead of failing here).
    for v in 0..n {
        let (p, ip) = (perm[v], iperm[v]);
        if p < 0 || ip < 0 || p as usize >= n || ip as usize >= n || perm[ip as usize] != v as i32
        {
            return Err(format!(
                "METIS perm/iperm are not inverses (rc={}, n={})",
                rc, n
            ));
        }
    }
    let swap = env::var("PSN_CHOL_METIS").map(|v| v == "iperm").unwrap_or(false);
    let q = if swap { iperm } else { perm };
    if !sp::is_bijection(&q) {
        return Err(format!(
            "METIS ordering is not a permutation (rc={}, n={})",
            rc, n
        ));
    }
    ORDER_CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, q.clone());
    Ok(q)
}
