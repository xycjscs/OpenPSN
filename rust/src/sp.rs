// Sparse structural helpers: input validation, symmetric CSR, elimination
// tree, postordering, and permutation of a CSC matrix.
//
// Conventions used throughout this crate:
//   * "original" numbering  = the CSC handed to `chol_factor` (Python side)
//   * "metis"  numbering    = after the fill-reducing ordering
//   * "internal" numbering  = after metis *and* the elimination-tree postorder
//   * `q[pos] = vertex`     = ordering array (position -> original vertex)
//   * `pos[v] = position`   = its inverse
//   * `map[internal] = original`  (used to permute RHS in / solutions out)

/// Validate the CSC triple that came in over the C ABI.
pub fn check_csc(n: usize, nz: usize, ap: &[i32], ai: &[i32], ax: &[f64]) -> Result<(), String> {
    if n == 0 {
        return Err("matrix dimension must be positive".into());
    }
    if ap.len() != n + 1 || ai.len() < nz || ax.len() < nz {
        return Err("sparse index arrays do not match (n, nnz)".into());
    }
    if ap[0] != 0 || (ap[n] as i64) != nz as i64 {
        return Err("indptr[0] != 0 or indptr[n] != nnz".into());
    }
    for k in 0..n {
        if ap[k] > ap[k + 1] {
            return Err("indptr is not nondecreasing".into());
        }
        for p in (ap[k] as usize)..(ap[k + 1] as usize) {
            let i = ai[p];
            if i < 0 || (i as usize) >= n {
                return Err(format!("row index {} out of range in column {}", i, k));
            }
        }
    }
    Ok(())
}

/// Symmetric adjacency (`A + A^T`, no self loops, no duplicates, sorted rows).
pub fn sym_csr(n: usize, ap: &[i32], ai: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut deg = vec![0usize; n];
    for col in 0..n {
        for p in (ap[col] as usize)..(ap[col + 1] as usize) {
            let i = ai[p] as usize;
            if i != col {
                deg[col] += 1;
                deg[i] += 1;
            }
        }
    }
    let mut xadj = vec![0i32; n + 1];
    let mut acc = 0usize;
    for v in 0..n {
        xadj[v] = acc as i32;
        acc += deg[v];
    }
    xadj[n] = acc as i32;
    let mut adj = vec![0i32; acc];
    let mut cur = xadj[..n].to_vec();
    for col in 0..n {
        for p in (ap[col] as usize)..(ap[col + 1] as usize) {
            let i = ai[p] as usize;
            if i == col {
                continue;
            }
            adj[cur[col] as usize] = i as i32;
            cur[col] += 1;
            adj[cur[i] as usize] = col as i32;
            cur[i] += 1;
        }
    }
    // sorted + deduplicated CSR.  Every undirected edge was inserted twice
    // (once from each endpoint's column), so this roughly halves the array.
    let mut compact: Vec<i32> = Vec::with_capacity(adj.len());
    let mut nxadj = vec![0i32; n + 1];
    for v in 0..n {
        let s = xadj[v] as usize;
        let e = xadj[v + 1] as usize;
        adj[s..e].sort_unstable();
        nxadj[v] = compact.len() as i32;
        let mut prev = -1i32;
        for &w in &adj[s..e] {
            if w != prev {
                compact.push(w);
                prev = w;
            }
        }
    }
    nxadj[n] = compact.len() as i32;
    (nxadj, compact)
}

/// `xadj/adj` relabelled by `pos` (vertex -> new label).
pub fn permute_csr(n: usize, xadj: &[i32], adj: &[i32], pos: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut deg = vec![0usize; n];
    for v in 0..n {
        deg[pos[v] as usize] = (xadj[v + 1] - xadj[v]) as usize;
    }
    let mut xadj2 = vec![0i32; n + 1];
    for v in 0..n {
        xadj2[v + 1] = xadj2[v] + deg[v] as i32;
    }
    let mut adj2 = vec![0i32; xadj2[n] as usize];
    for v in 0..n {
        let dst = pos[v] as usize;
        let s = xadj2[dst] as usize;
        for (t, w) in adj[(xadj[v] as usize)..(xadj[v + 1] as usize)]
            .iter()
            .enumerate()
        {
            adj2[s + t] = pos[*w as usize];
        }
        adj2[s..(s + deg[dst])].sort_unstable();
    }
    (xadj2, adj2)
}

/// Elimination tree of the symmetric pattern; `parent[v] > v`, `-1` at a root.
pub fn etree(n: usize, xadj: &[i32], adj: &[i32]) -> Vec<i32> {
    let mut parent = vec![-1i32; n];
    let mut mark = vec![-1i32; n];
    for col in 0..n {
        let c = col as i32;
        for idx in (xadj[col] as usize)..(xadj[col + 1] as usize) {
            let mut v = adj[idx] as usize;
            if (v as i32) >= c {
                continue; // lower triangle only
            }
            while mark[v] != c {
                mark[v] = c;
                if parent[v] < 0 {
                    parent[v] = c;
                    break;
                }
                v = parent[v] as usize;
            }
        }
    }
    parent
}

/// Postorder of the elimination tree: `post[k] = node`, `ipost[node] = k`.
pub fn postorder(n: usize, parent: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut head = vec![-1i32; n];
    let mut sib = vec![-1i32; n];
    for v in 0..n {
        let p = parent[v];
        if p >= 0 {
            let p = p as usize;
            sib[v] = head[p];
            head[p] = v as i32;
        }
    }
    let mut post: Vec<i32> = Vec::with_capacity(n);
    let mut ipost = vec![0i32; n];
    // iterative depth-first with an explicit child cursor
    let mut stack: Vec<(usize, i32)> = Vec::with_capacity(64);
    for root in 0..n {
        if parent[root] >= 0 {
            continue;
        }
        stack.clear();
        stack.push((root, head[root]));
        while !stack.is_empty() {
            let (_v, cur) = stack.last().copied().unwrap();
            if cur < 0 {
                let (v, _) = stack.pop().unwrap();
                ipost[v] = post.len() as i32;
                post.push(v as i32);
            } else {
                let c = cur as usize;
                stack.last_mut().unwrap().1 = sib[c];
                stack.push((c, head[c]));
            }
        }
    }
    (post, ipost)
}

/// `A2 = A2_internal`: permuted CSC, BOTH triangles kept, rows sorted.
///
/// `colmap[internal col] = original col` and `rowmap[original row] = internal
/// row` — i.e. `rowmap` is the INVERSE of `colmap`.  (They cannot be the same
/// array unless the permutation is the identity: passing `map` for both silently
/// puts every diagonal entry on the wrong row and the factorization then dies on
/// a zero pivot.)
pub fn permute_csc(
    n: usize,
    ap: &[i32],
    ai: &[i32],
    ax: &[f64],
    colmap: &[i32],
    rowmap: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut p = vec![0i32; n + 1];
    for k in 0..n {
        let c = colmap[k] as usize;
        p[k + 1] = p[k] + (ap[c + 1] - ap[c]);
    }
    let nnz = p[n] as usize;
    let mut i = vec![0i32; nnz];
    let mut x = vec![0.0f64; nnz];
    let mut tmp: Vec<(i32, f64)> = Vec::new();
    for k in 0..n {
        let c = colmap[k] as usize;
        let s = p[k] as usize;
        tmp.clear();
        tmp.extend(
            (ap[c] as usize..ap[c + 1] as usize)
                .map(|q| (rowmap[ai[q] as usize], ax[q])),
        );
        tmp.sort_unstable_by_key(|e| e.0);
        for (t, (ri, rv)) in tmp.iter().enumerate() {
            i[s + t] = *ri;
            x[s + t] = *rv;
        }
    }
    (p, i, x)
}

pub fn is_bijection(v: &[i32]) -> bool {
    let n = v.len();
    let mut seen = vec![false; n];
    for &x in v {
        if x < 0 || x as usize >= n || seen[x as usize] {
            return false;
        }
        seen[x as usize] = true;
    }
    true
}
