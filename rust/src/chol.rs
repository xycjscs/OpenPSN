// Left-looking sparse LL^T factorization with a dense accumulator.
//
// For column k the identity A(:,k) = sum_{j<=k} L(k,j) L(:,j) says that
// xi = L^-1 A(:,k) holds L(k,j) in xi[j]: the discovered set is the support of
// ROW k, and every entry of the NEW column k (rows i > k) is read off the same
// dense accumulator.  So the traversal needs the row view of L as well:
//
//   for j in row_k_of_L (ascending, so w[j] is final when used):
//       s = w[j] / L(j,j)                       // = L(k,j)
//       for (i, v) in column j with i > j: w[i] -= s * v
//   L(k,k) = sqrt(w[k]);  L(i,k) = w[i] / L(k,k) for the marked rows i > k
//
// Work per column is sum_{j in row_k} nnz(L(:,j)) — the same count a
// left-looking SimplicialLLT does, so factor times and factor nnz stay
// comparable with the C++ bridge.  Storage of L: column-major, off-diagonal
// rows ascending, diagonal LAST (the layout `panels` repacks).
use crate::sp;

pub struct L {
    pub lp: Vec<i32>,
    pub li: Vec<i32>,
    pub lx: Vec<f64>,
}

/// Row view of L, built incrementally as columns are created (linked lists; a
/// row's columns come out descending, which `gather` reverses to ascending).
struct RowView {
    head: Vec<i32>,
    next: Vec<i32>,
    col: Vec<i32>,
}

impl RowView {
    fn new(n: usize) -> Self {
        RowView {
            head: vec![-1i32; n],
            next: Vec::new(),
            col: Vec::new(),
        }
    }
    #[inline]
    fn push(&mut self, row: usize, c: i32) {
        let idx = self.col.len() as i32;
        self.col.push(c);
        self.next.push(self.head[row]);
        self.head[row] = idx;
    }
    /// Columns of `row`, ascending.
    fn gather(&self, row: usize, out: &mut Vec<i32>) {
        out.clear();
        let mut p = self.head[row];
        while p >= 0 {
            out.push(self.col[p as usize]);
            p = self.next[p as usize];
        }
        out.reverse(); // strictly descending list -> strictly ascending
    }
}

/// Factor the (already permuted, full, row-sorted) CSC `A` as `L L^T`.
///
/// Returns `Err((2, col, pivot))` (the `Eigen::NumericalIssue` code) when a
/// pivot is not strictly positive.
pub fn factor(n: usize, ap: &[i32], ai: &[i32], ax: &[f64]) -> Result<L, (usize, f64)> {
    let mut lp = vec![0i32; n + 1];
    let nnz_a = ap[n] as usize;
    let cap = (4 * nnz_a).max(8 * n).max(1024);
    let mut li: Vec<i32> = Vec::with_capacity(cap);
    let mut lx: Vec<f64> = Vec::with_capacity(cap);
    let mut w = vec![0.0f64; n]; // dense accumulator, zero outside `touched`
    let mut mark = vec![0u8; n];
    let mut touched: Vec<usize> = Vec::with_capacity(256);
    let mut rows_k: Vec<i32> = Vec::with_capacity(64);
    let mut newrows: Vec<usize> = Vec::with_capacity(64);
    let mut rv = RowView::new(n);

    for k in 0..n {
        lp[k] = li.len() as i32;
        touched.clear();
        // ---- seed: w = A(:,k); rows i > k seed the new column too ----------
        for p in (ap[k] as usize)..(ap[k + 1] as usize) {
            let i = ai[p] as usize;
            w[i] += ax[p];
            if mark[i] == 0 {
                mark[i] = 1;
                touched.push(i);
            }
        }
        // ---- eliminate the columns appearing in row k ----------------------
        rv.gather(k, &mut rows_k);
        for t in 0..rows_k.len() {
            let j = rows_k[t] as usize;
            let dj = (lp[j + 1] - 1) as usize; // diagonal is stored last
            let s = w[j] / lx[dj];
            if s == 0.0 {
                continue; // no numeric contribution and no spurious fill
            }
            for p in (lp[j] as usize)..dj {
                let i = li[p] as usize; // ascending, all > j
                w[i] -= s * lx[p];
                if mark[i] == 0 {
                    mark[i] = 1;
                    touched.push(i);
                }
            }
        }
        // ---- pivot --------------------------------------------------------
        let piv2 = w[k];
        if !(piv2 > 0.0) {
            for &i in touched.iter() {
                w[i] = 0.0;
                mark[i] = 0;
            }
            return Err((k, piv2));
        }
        let piv = piv2.sqrt();
        // ---- the new column: marked rows i > k, ascending -----------------
        newrows.clear();
        for &i in touched.iter() {
            if i > k {
                newrows.push(i);
            }
        }
        newrows.sort_unstable();
        let need = newrows.len() + 1;
        if li.len() + need > li.capacity() {
            li.reserve(need.max(li.len() / 4));
            lx.reserve(need.max(lx.len() / 4));
        }
        for &i in newrows.iter() {
            li.push(i as i32);
            lx.push(w[i] / piv);
            rv.push(i, k as i32); // L(i,k) != 0 -> column k belongs to row i
        }
        li.push(k as i32);
        lx.push(piv);
        lp[k + 1] = li.len() as i32;
        for &i in touched.iter() {
            w[i] = 0.0;
            mark[i] = 0;
        }
    }
    li.shrink_to_fit();
    lx.shrink_to_fit();
    Ok(L { lp, li, lx })
}

/// Ordering + numeric pipeline shared by `chol_factor`.
///
/// `ordering != 0` asks METIS for a fill-reducing permutation (hard failure if
/// METIS cannot be reached); `ordering == 0` factors the matrix as given.  The
/// elimination-tree postorder is applied on top either way (locality only — it
/// does not change `nnz(L)`).  Returns `(L, map)` with
/// `map[internal row] = original row`.
pub fn analyze_and_factor(
    n: usize,
    ap: &[i32],
    ai: &[i32],
    ax: &[f64],
    ordering: i32,
) -> Result<(L, Vec<i32>), (String, i32)> {
    let mut q: Vec<i32> = (0..n).map(|v| v as i32).collect();
    let (xadj0, adj0) = sp::sym_csr(n, ap, ai);
    if ordering != 0 {
        q = crate::metis::node_nd(n, &xadj0, &adj0).map_err(|m| (m, -3))?;
    }
    let mut pos = vec![0i32; n];
    for (p, v) in q.iter().enumerate() {
        pos[*v as usize] = p as i32;
    }
    // elimination tree + postorder in the metis numbering
    let (xadj1, adj1) = sp::permute_csr(n, &xadj0, &adj0, &pos);
    drop((xadj0, adj0));
    let parent = sp::etree(n, &xadj1, &adj1);
    let (post, _ipost) = sp::postorder(n, &parent);
    drop((xadj1, adj1, parent));
    // composite internal -> original map
    let map: Vec<i32> = post.iter().map(|&v| q[v as usize]).collect();
    if !sp::is_bijection(&map) {
        return Err(("permutation is not a bijection".into(), -3));
    }
    drop((post, q, pos));
    // columns are gathered by internal -> original, rows by original -> internal
    let mut inv = vec![0i32; n];
    for (k, v) in map.iter().enumerate() {
        inv[*v as usize] = k as i32;
    }
    let (p2, i2, x2) = sp::permute_csc(n, ap, ai, ax, &map, &inv);
    drop(inv);
    let l = factor(n, &p2, &i2, &x2).map_err(|(k, p)| {
        (
            format!(
                "matrix is not positive definite: pivot {} at internal column {} \
                 (original row {}), status 2",
                p,
                k,
                map[k]
            ),
            2,
        )
    })?;
    drop((p2, i2, x2));
    Ok((l, map))
}
