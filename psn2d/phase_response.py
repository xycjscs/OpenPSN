"""Static condensation of finite-sector PSN cell-interior moments."""

import numpy as np
from scipy.sparse import coo_matrix


class NodeResponse:
    """Solve for shared traces and reconstruct the element-local moments."""

    def __init__(self, psn):
        mesh = psn.mesh
        reduced = psn._phase_reduction
        if reduced is None:
            centers = mesh.cells[:, 4]
            centers = np.unique(np.r_[centers, mesh.mirror[centers]])
            sizes = [mesh.bilinear_size if psn._bilinear_mode(mode) else mesh.size
                     for mode in range(psn.angular_modes)]
        else:
            centers = np.unique(reduced.nodes[0][:, 4])
            sizes = reduced.shapes

        offsets = np.cumsum([0, *sizes])
        quadratic = [offsets[mode] for mode in range(psn.angular_modes)
                     if not psn._bilinear_mode(mode)]
        self.centers = centers[:, None] + quadratic
        self.interior = self.centers.ravel()
        self.size = int(offsets[-1])
        keep = np.ones(self.size, dtype=bool)
        keep[self.interior] = False
        self.trace = np.flatnonzero(keep)

    def factor(self, matrix, factor):
        trace, interior = self.trace, self.interior
        count, order = self.centers.shape
        row = np.repeat(self.centers, order, axis=1).ravel()
        col = np.tile(self.centers, (1, order)).ravel()
        blocks = np.asarray(matrix[row, col]).reshape(count, order, order)
        inverse = np.linalg.inv(blocks)

        # Interior shapes vanish on every element face. Their matrix has one
        # independent dense block per cell and sign, or per reflection orbit.
        ni = interior.size
        ids = np.arange(ni, dtype=np.int32).reshape(count, order)
        inverse_sparse = coo_matrix(
            (inverse.ravel(), (np.repeat(ids, order, axis=1).ravel(), np.tile(ids, (1, order)).ravel())),
            shape=(ni, ni),
        ).tocsr()
        coupling = matrix[interior, :][:, trace]
        response = (inverse_sparse @ coupling).tocsr()
        trace_matrix = matrix[trace, :][:, trace].tocsc()
        schur = trace_matrix - coupling.T @ response
        if schur.nnz != trace_matrix.nnz:
            # Eliminating a cell center only couples DOFs already on that
            # cell. Keep structural zeros for the shared symbolic factor,
            # even when one particular angle cancels a matrix entry.
            cols = np.repeat(np.arange(trace.size), np.diff(trace_matrix.indptr))
            trace_matrix.data[:] = np.asarray(schur[trace_matrix.indices, cols]).ravel()
            schur = trace_matrix
        solve_trace, size = factor(schur)
        size += inverse.nbytes + response.data.nbytes + response.indices.nbytes + response.indptr.nbytes

        def solve(rhs):
            cell_rhs = rhs[interior]
            face = solve_trace(rhs[trace] - response.T @ cell_rhs)
            center = (inverse @ cell_rhs.reshape(count, order, 1)).ravel() - response @ face
            result = np.empty(self.size)
            result[trace] = face
            result[interior] = center
            return result

        return solve, size
