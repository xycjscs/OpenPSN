// Packed supernodal panels + the triangular solve kernels.
//
// After the column-wise factorization the columns of L are merged into
// supernodes (maximal runs of columns whose row sets satisfy
// rows(j) == rows(j-1) \ {j-1}) and every column block of at most `SMAX`
// columns is stored as one DENSE, ROW-MAJOR h x s block:
//
//   vals[val_base + r*s + c] = L(rows[row_base + r],  col + c)
//
// with zeros in the structurally empty (strictly upper) corner.  Consequence
// for the solve, which is what this whole file exists for:
//
//   * the numeric block is traversed sequentially, one cache line per row;
//   * the row-index list is touched once per panel, not once per entry;
//   * the inner product per row is `s` contiguous multiplies against a
//     register-resident vector, i.e. the indirect indexing that dominates a
//     simplicial solve is amortised over `s` columns.
//
// Both substitution passes are read-only with respect to `Panels`, so a
// forked pool can share the factor COW.
use crate::chol::L;

pub const META: usize = 5;
/// Hard cap on a panel's column-block width (compile-time array bound).
pub const SMAX: usize = 16;

pub struct Panels {
    /// per panel: col, s, row_base, h, val_base
    meta: Vec<i64>,
    rows: Vec<i32>,
    vals: Vec<f64>,
}

impl Panels {
    #[inline]
    pub fn npanels(&self) -> usize {
        self.meta.len() / META
    }
    /// Bytes actually held by the factor (excluding the handle itself).
    pub fn bytes(&self) -> usize {
        (self.meta.len() * 8) + (self.rows.len() * 4) + (self.vals.len() * 8)
    }
    pub fn panel_rows(&self) -> usize {
        self.rows.len()
    }
    pub fn stored_vals(&self) -> usize {
        self.vals.len()
    }
    /// Raw views, for the debug driver and tests only.
    pub fn parts(&self) -> (&[i64], &[i32], &[f64]) {
        (&self.meta, &self.rows, &self.vals)
    }
}

#[inline]
fn same_supernode(l: &L, j: usize) -> bool {
    // rows(j) == rows(j-1) \ {j-1}, with L stored off-diagonals ascending and
    // the diagonal LAST:  A = offdiag(j-1) must equal offdiag(j) plus a
    // leading entry equal to j.
    let (a0, a1) = (l.lp[j - 1] as usize, l.lp[j] as usize);
    let (b0, b1) = (l.lp[j] as usize, l.lp[j + 1] as usize);
    let off_a = a1 - a0 - 1;
    let off_b = b1 - b0 - 1;
    if off_a != off_b + 1 || l.li[a0] != j as i32 {
        return false;
    }
    l.li[a0 + 1..a1 - 1] == l.li[b0..b1 - 1]
}

fn pack_panel(
    l: &L,
    cb: usize,
    s: usize,
    meta: &mut Vec<i64>,
    rows: &mut Vec<i32>,
    vals: &mut Vec<f64>,
    blk: &mut Vec<f64>,
) -> Result<(), String> {
    let bad = |m: &str| Err(format!("{} (column {})", m, cb));
    let lc = l.lp[cb] as usize;
    let lc1 = l.lp[cb + 1] as usize;
    let off = lc1 - lc - 1;
    if off + 1 < s {
        return bad("supernode row set smaller than its column block");
    }
    for t in 0..(s - 1) {
        if l.li[lc + t] != (cb + 1 + t) as i32 {
            return bad("non-contiguous diagonal block");
        }
    }
    let row_base = rows.len();
    let val_base = vals.len();
    rows.reserve(off + 2);
    for t in 0..s {
        rows.push((cb + t) as i32);
    }
    rows.extend_from_slice(&l.li[lc + (s - 1)..lc1 - 1]);
    let h = rows.len() - row_base;
    debug_assert_eq!(h, off + 1);

    blk.clear();
    blk.resize(h * s, 0.0);
    for c in cb..(cb + s) {
        let cc = c - cb;
        let cs = l.lp[c] as usize;
        let ce = l.lp[c + 1] as usize;
        if l.li[ce - 1] != c as i32 {
            return bad("diagonal entry is not last in its column");
        }
        blk[cc * s + cc] = l.lx[ce - 1];
        let mut rp = cc + 1;
        for p in cs..(ce - 1) {
            let want = if rp < s {
                (cb + rp) as i32
            } else {
                l.li[lc + rp - 1]
            };
            if want != l.li[p] {
                return bad("panel row/column structure mismatch");
            }
            blk[rp * s + cc] = l.lx[p];
            rp += 1;
        }
        if rp != h {
            return bad("panel row count mismatch");
        }
    }
    vals.extend_from_slice(&blk[.. h * s]);
    meta.extend_from_slice(&[
        cb as i64,
        s as i64,
        row_base as i64,
        h as i64,
        val_base as i64,
    ]);
    Ok(())
}

/// Merge the columns of `L` into column blocks of at most `sblock` columns.
pub fn build(l: &L, sblock: usize) -> Result<Panels, String> {
    let sblock = sblock.clamp(1, SMAX);
    let n = l.lp.len() - 1;
    let mut meta: Vec<i64> = Vec::with_capacity(n / META);
    let mut rows: Vec<i32> = Vec::with_capacity(l.li.len());
    let mut vals: Vec<f64> = Vec::with_capacity(l.lx.len());
    let mut blk: Vec<f64> = Vec::new();
    let mut k = 0usize;
    while k < n {
        let mut j = k + 1;
        while j < n && same_supernode(l, j) {
            j += 1;
        }
        let mut cb = k;
        while cb < j {
            let s = (j - cb).min(sblock);
            pack_panel(l, cb, s, &mut meta, &mut rows, &mut vals, &mut blk)?;
            cb += s;
        }
        k = j;
    }
    meta.shrink_to_fit();
    rows.shrink_to_fit();
    vals.shrink_to_fit();
    Ok(Panels { meta, rows, vals })
}

// --------------------------------------------------------------------------- #
// solve kernels
// --------------------------------------------------------------------------- #
//
// Both kernels dispatch their panel-row loop on a CONSTANT column width.  At
// the default `sblock=8` an M16_S2 panel averages 15 rows x 2.7 columns, so
// with `s` only known at run time the inner loop is a real loop — a compare and
// two pointer bumps per multiply-add, and no independent work to interleave —
// and that bookkeeping, not the arithmetic, is what the solve pays for (the
// solve is throughput-bound: K right-hand sides cost 1.00/1.74/2.68/3.87x for
// K=1/2/3/4, where a latency-bound kernel would sit near 1.0).
//
// Specialising the width changes no arithmetic: the forward pass additionally
// runs four rows at once, which cannot matter because panel rows share no
// accumulator — each `d` still sums `c` in ascending order.  The backward pass
// deliberately does NOT get the same interleave: its per-column `acc[c]` is a
// reduction over rows, so two rows in flight would reorder a sum.  Verified by
// comparing solutions against the pre-specialisation library byte for byte.

macro_rules! rows_by_s {
    ($s:expr, $f:ident, $h:expr, $rb:expr, $vb:expr, $rows:expr, $vals:expr, $u:expr, $t:expr, $ld:expr) => {
        match $s {
            1 => $f::<1, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            2 => $f::<2, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            3 => $f::<3, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            4 => $f::<4, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            5 => $f::<5, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            6 => $f::<6, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            7 => $f::<7, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            8 => $f::<8, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
            _ => $f::<0, K>($h, $rb, $vb, $rows, $vals, $u, $t, $ld, $s),
        }
    };
}

/// Forward-pass row loop.  `S = 0` means "width unknown at compile time" (only
/// reachable through `PSN_CHOL_S > 8`, the old code path).
#[inline(always)]
unsafe fn fwd_rows<const S: usize, const K: usize>(
    h: usize,
    rb: usize,
    vb: usize,
    rows: &[i32],
    vals: &[f64],
    u: [[f64; SMAX]; K],
    t: *mut f64,
    ld: usize,
    s: usize,
) {
    let w = if S == 0 { s } else { S };
    let mut r = w;
    while r + 4 <= h {
        let b = vb + r * w;
        for q in 0..K {
            let up = &u[q];
            let (mut d0, mut d1, mut d2, mut d3) = (0.0, 0.0, 0.0, 0.0);
            if S == 0 {
                for c in 0..w {
                    let x = *up.get_unchecked(c);
                    d0 += *vals.get_unchecked(b + c) * x;
                    d1 += *vals.get_unchecked(b + w + c) * x;
                    d2 += *vals.get_unchecked(b + 2 * w + c) * x;
                    d3 += *vals.get_unchecked(b + 3 * w + c) * x;
                }
            } else {
                for c in 0..S {
                    let x = *up.get_unchecked(c);
                    d0 += *vals.get_unchecked(b + c) * x;
                    d1 += *vals.get_unchecked(b + S + c) * x;
                    d2 += *vals.get_unchecked(b + 2 * S + c) * x;
                    d3 += *vals.get_unchecked(b + 3 * S + c) * x;
                }
            }
            let tq = t.add(q * ld);
            *tq.add(*rows.get_unchecked(rb + r) as usize) -= d0;
            *tq.add(*rows.get_unchecked(rb + r + 1) as usize) -= d1;
            *tq.add(*rows.get_unchecked(rb + r + 2) as usize) -= d2;
            *tq.add(*rows.get_unchecked(rb + r + 3) as usize) -= d3;
        }
        r += 4;
    }
    while r < h {
        let idx = *rows.get_unchecked(rb + r) as usize;
        let b = vb + r * w;
        for q in 0..K {
            let mut d = 0.0;
            for c in 0..w {
                d += *vals.get_unchecked(b + c) * u[q][c];
            }
            *t.add(q * ld + idx) -= d;
        }
        r += 1;
    }
}

/// Backward-pass row loop: the same constant-width treatment, but the rows stay
/// strictly sequential because `acc[q][c]` accumulates over them.
#[inline(always)]
unsafe fn bwd_rows<const S: usize, const K: usize>(
    h: usize,
    rb: usize,
    vb: usize,
    rows: &[i32],
    vals: &[f64],
    acc: &mut [[f64; SMAX]; K],
    t: *const f64,
    ld: usize,
    s: usize,
) {
    let w = if S == 0 { s } else { S };
    let mut r = w;
    while r < h {
        let idx = *rows.get_unchecked(rb + r) as usize;
        let b = vb + r * w;
        for q in 0..K {
            let tv = *t.add(q * ld + idx);
            for c in 0..w {
                acc[q][c] += *vals.get_unchecked(b + c) * tv;
            }
        }
        r += 1;
    }
}

/// `t <- L^-1 t` for `K` right-hand sides stored as `K` contiguous blocks of
/// `ld` doubles (`t[q*ld + i]`).
unsafe fn forward<const K: usize>(
    ld: usize,
    meta: &[i64],
    rows: &[i32],
    vals: &[f64],
    t: *mut f64,
) {
    let np = meta.len() / META;
    let mut u = [[0.0f64; SMAX]; K];
    for p in 0..np {
        let b = p * META;
        let col = *meta.get_unchecked(b) as usize;
        let s = *meta.get_unchecked(b + 1) as usize;
        let rb = *meta.get_unchecked(b + 2) as usize;
        let h = *meta.get_unchecked(b + 3) as usize;
        let vb = *meta.get_unchecked(b + 4) as usize;
        for q in 0..K {
            let tq = t.add(q * ld);
            for c in 0..s {
                u[q][c] = *tq.add(col + c);
            }
            for c in 0..s {
                let mut acc = u[q][c];
                let r0 = vb + c * s;
                for cp in 0..c {
                    acc -= *vals.get_unchecked(r0 + cp) * u[q][cp];
                }
                u[q][c] = acc / *vals.get_unchecked(r0 + c);
            }
            for c in 0..s {
                *tq.add(col + c) = u[q][c];
            }
        }
        rows_by_s!(s, fwd_rows, h, rb, vb, rows, vals, u, t, ld);
    }
}

/// `t <- L^-T t`, same layout, panels in reverse.
unsafe fn backward<const K: usize>(
    ld: usize,
    meta: &[i64],
    rows: &[i32],
    vals: &[f64],
    t: *mut f64,
) {
    let np = meta.len() / META;
    let mut acc = [[0.0f64; SMAX]; K];
    for p in (0..np).rev() {
        let b = p * META;
        let col = *meta.get_unchecked(b) as usize;
        let s = *meta.get_unchecked(b + 1) as usize;
        let rb = *meta.get_unchecked(b + 2) as usize;
        let h = *meta.get_unchecked(b + 3) as usize;
        let vb = *meta.get_unchecked(b + 4) as usize;
        for q in 0..K {
            for c in 0..s {
                acc[q][c] = 0.0;
            }
        }
        match s {
            1 => bwd_rows::<1, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            2 => bwd_rows::<2, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            3 => bwd_rows::<3, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            4 => bwd_rows::<4, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            5 => bwd_rows::<5, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            6 => bwd_rows::<6, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            7 => bwd_rows::<7, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            8 => bwd_rows::<8, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
            _ => bwd_rows::<0, K>(h, rb, vb, rows, vals, &mut acc, t, ld, s),
        }
        for q in 0..K {
            let tq = t.add(q * ld);
            let mut c = s;
            while c > 0 {
                c -= 1;
                let mut sum = acc[q][c];
                for cp in (c + 1)..s {
                    sum += *vals.get_unchecked(vb + cp * s + c) * *tq.add(col + cp);
                }
                *tq.add(col + c) =
                    (*tq.add(col + c) - sum) / *vals.get_unchecked(vb + c * s + c);
            }
        }
    }
}

impl Panels {
    /// One `L L^T` solve for `K` right-hand sides, in place, read-only factor.
    #[inline]
    pub unsafe fn solve<const K: usize>(&self, ld: usize, t: *mut f64) {
        forward::<K>(ld, &self.meta, &self.rows, &self.vals, t);
        backward::<K>(ld, &self.meta, &self.rows, &self.vals, t);
    }

    /// `K` outside the specialised widths: `K` independent single-RHS solves.
    pub unsafe fn solve_generic(&self, ld: usize, t: *mut f64, k: usize) {
        match k {
            1 => self.solve::<1>(ld, t),
            2 => self.solve::<2>(ld, t),
            3 => self.solve::<3>(ld, t),
            4 => self.solve::<4>(ld, t),
            _ => {
                for q in 0..k {
                    self.solve::<1>(ld, t.add(q * ld));
                }
            }
        }
    }
}
