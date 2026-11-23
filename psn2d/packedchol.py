"""Packed Cholesky preconditioning for shared PSN face-current blocks."""
import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def load_kernel():
    """Bind the packed entry points in the shared Rust Cholesky library."""
    from rustchol import load_rust_chol

    lib = load_rust_chol()
    if not hasattr(lib, "packed_factor"):
        raise RuntimeError("Rust kernel lacks packed Cholesky support; rebuild rust/build.sh")
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    lib.packed_factor.argtypes = [size, size, pointer, pointer, pointer,
                                  ctypes.c_uint32, pointer]
    lib.packed_factor.restype = pointer
    lib.packed_free.argtypes = [pointer]
    lib.packed_free.restype = None
    lib.packed_solve.argtypes = [size, size] + [pointer] * 6 + [
        ctypes.c_double, ctypes.c_int32, ctypes.POINTER(ctypes.c_double)]
    lib.packed_solve.restype = ctypes.c_int32
    return lib


def encode_rows(matrix):
    """Store the original FP64 CSR rows as unique relative-index templates."""
    lengths = np.diff(matrix.indptr)
    width = int(lengths.max())
    dtype = np.dtype([("offsets", np.int32, (width,)),
                      ("values", np.float64, (width,))])
    rows = np.zeros(matrix.shape[0], dtype=dtype)
    for slot in range(width):
        present = np.flatnonzero(lengths > slot)
        index = matrix.indptr[present] + slot
        rows["offsets"][present, slot] = matrix.indices[index] - present
        rows["values"][present, slot] = matrix.data[index]
    templates, codes = np.unique(rows.view(f"V{dtype.itemsize}"), return_inverse=True)
    templates = templates.view(dtype)
    return (width, np.ascontiguousarray(codes, dtype=np.uint32),
            np.ascontiguousarray(templates["offsets"]),
            np.ascontiguousarray(templates["values"]))


class _PackedFactor:
    def __init__(self, matrix, lib):
        self.lib = lib
        self.handle = None
        self.size = matrix.shape[0]
        if max(self.size, matrix.nnz) > np.iinfo(np.int32).max:
            raise ValueError("packed factor input exceeds the int32 index range")
        matrix.sort_indices()
        self.width, self.codes, self.offsets, self.values = encode_rows(matrix)
        csc = matrix.tocsc()
        csc.sort_indices()
        indptr = np.ascontiguousarray(csc.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(csc.indices, dtype=np.int32)
        values = np.ascontiguousarray(csc.data, dtype=np.float64)
        info = np.zeros(4, dtype=np.uint64)
        self.handle = lib.packed_factor(
            self.size, csc.nnz, indptr.ctypes.data, indices.ctypes.data,
            values.ctypes.data, 16, info.ctypes.data)
        if not self.handle:
            raise RuntimeError("packed Cholesky factor construction failed")
        self.factor_nnz = int(info[0])
        self.pool_bytes = int(info[3]) + sum(
            array.nbytes for array in (self.codes, self.offsets, self.values))

    def solve(self, rhs):
        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        if rhs.shape != (self.size,):
            raise ValueError(f"expected packed RHS with shape ({self.size},)")
        out = np.empty_like(rhs)
        residual = ctypes.c_double()
        steps = self.lib.packed_solve(
            self.size, self.width, self.codes.ctypes.data,
            self.offsets.ctypes.data, self.values.ctypes.data,
            self.handle, rhs.ctypes.data, out.ctypes.data,
            8e-13, 45, ctypes.byref(residual))
        if steps < 0:
            raise RuntimeError(f"packed Cholesky PCG residual "
                               f"{residual.value:.2e} after {-steps} steps")
        return out

    def __del__(self):
        if self.handle:
            self.lib.packed_free(self.handle)


class PackedSharedChol:
    def __init__(self, A, d, pairs, ordering=1, check=True, *, lib, workers):
        if ordering != 1:
            raise ValueError("packed Cholesky requires METIS ordering")
        self.d = d
        self.pairs = pairs
        self.workers = workers

        def make_pair(pair):
            columns, mirror, sign = pair
            block = A[columns, :][:, columns].multiply(d[columns, None]).tocsr()
            count, asymmetry, mismatch = 0, 0.0, 0.0
            if check:
                twin = A[mirror, :][:, mirror].multiply(
                    (d[mirror] * sign)[:, None]).multiply(sign[None, :])
                magnitude = float(abs(block).max())
                count = block.nnz + twin.nnz
                delta = block - block.T
                asymmetry = float(abs(delta).max() / magnitude) if delta.nnz else 0.0
                if asymmetry > 1e-12:
                    raise ValueError(f"nonsymmetric angle-pair block: {asymmetry:.2e}")
                delta = block - twin
                mismatch = float(abs(delta).max() / magnitude) if delta.nnz else 0.0
                if mismatch > 1e-12:
                    raise ValueError(f"angle-pair symmetry mismatch: {mismatch:.2e}")
                del twin, delta
            return _PackedFactor(block, lib), count, asymmetry, mismatch

        with ThreadPoolExecutor(max_workers=workers) as pool:
            built = list(pool.map(make_pair, pairs))
        self.factors, counts, asymmetries, self.symmetry_errors = zip(*built)
        if check and sum(counts) != A.nnz:
            raise ValueError("found coupling outside the mirror-pair blocks")
        self.scaled_asymmetry = max(asymmetries)
        self.factor_nnz = sum(factor.factor_nnz for factor in self.factors)
        self.pool_bytes = sum(factor.pool_bytes for factor in self.factors)

    def solve(self, data):
        rhs = self.d * data
        out = np.empty_like(rhs)

        def solve_pair(job):
            factor, (columns, mirror, sign) = job
            a, b = rhs[columns], sign * rhs[mirror]
            x = factor.solve(a)
            reuse = np.linalg.norm(a - b) <= 1e-14 * np.linalg.norm(a)
            y = x if reuse else factor.solve(b)
            out[columns], out[mirror] = x, sign * y

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for _ in pool.map(solve_pair, zip(self.factors, self.pairs)):
                pass
        return out
