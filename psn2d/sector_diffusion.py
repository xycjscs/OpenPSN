"""Even-parity phase-space diffusion with quadratic spatial nodes.

For isotropic sources, eliminating the odd flux gives
    -div(omega omega^T / Sigma_t grad u) + Sigma_t u = q.
Take the weak form in each angular sector with a Gauss nodal basis. Quadrature
diagonalizes the angular mass, streaming tensor and vacuum term together.
The finite-sector closure integrates the streaming tensor and vacuum term
against an angular basis local to each cell.
In both closures, reflected fields share spatial trace degrees of freedom
at specular faces, so their normal currents sum to zero there.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import splu

from . import numa
from .ritz import keff_ritz
from .solver import PSN2D


def _quadratic_forms():
    """Operators on the unit interval at the three Lagrange nodes 0, 1/2, 1."""
    z, w = leggauss(3)
    values = np.column_stack((z * (z - 1) / 2, 1 - z * z, z * (z + 1) / 2))
    grad = np.column_stack((2 * z - 1, -4 * z, 2 * z + 1))
    mass = values.T @ ((w / 2)[:, None] * values)
    stiff = grad.T @ ((w / 2)[:, None] * grad)
    mix = values.T @ ((w / 2)[:, None] * grad)
    project = legvander(z, 2).T @ ((w / 2)[:, None] * values)
    project *= np.sqrt(2 * np.arange(3) + 1)[:, None]
    return mass, stiff, mix, project


_MASS, _STIFF, _MIX, _PROJECT = _quadratic_forms()
_CELL_MASS = np.kron(_MASS, _MASS)
_GRAD_X = np.kron(_MASS, _STIFF)
_GRAD_Y = np.kron(_STIFF, _MASS)
_CROSS = np.kron(_MIX.T, _MIX) + np.kron(_MIX, _MIX.T)
_MOMENTS = np.kron(_PROJECT, _PROJECT)
_LINEAR = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
_BILINEAR = np.kron(_LINEAR, _LINEAR)
_CORNERS = [0, 2, 6, 8]


def _restricted_forms(row_bilinear, col_bilinear):
    cell = (_CELL_MASS, _GRAD_X, _GRAD_Y, _CROSS)
    face = _MASS
    if row_bilinear:
        cell = tuple(_BILINEAR.T @ form for form in cell)
        face = _LINEAR.T @ face
    if col_bilinear:
        cell = tuple(form @ _BILINEAR for form in cell)
        face = face @ _LINEAR
    return (*cell, face)


_FORMS = {pair: _restricted_forms(*pair) for pair in product((False, True), repeat=2)}


class _PhasePool:
    """Run each prefactor and its sweeps on the same NUMA node when enabled."""

    def __init__(self, workers, homes):
        if not homes:
            self.pools = [ThreadPoolExecutor(max_workers=workers)]
            return
        per_node, extra = divmod(workers, len(homes))
        self.pools = [ThreadPoolExecutor(max_workers=min(len(cpus), per_node + (i < extra)),
                                         initializer=self._place, initargs=(node, cpus))
                      for i, (node, cpus) in enumerate(homes)]

    @staticmethod
    def _place(node, cpus):
        numa.set_cpu(cpus)
        numa.set_mem(node)

    def map(self, fn, jobs, offset=0):
        if len(self.pools) == 1:
            return self.pools[0].map(fn, jobs)
        futures = [self.pools[index % len(self.pools)].submit(fn, job)
                   for index, job in enumerate(jobs, start=offset)]

        def ordered():
            for index, future in enumerate(futures):
                yield future.result()
                futures[index] = None

        return ordered()

    def shutdown(self):
        for pool in self.pools:
            pool.shutdown()


def _directions(sectors, azimuth_degree, mu, polar_weights, transpose_symmetry):
    """Return one positive-x, positive-y member and weight per mirror pair."""
    z, w = leggauss(azimuth_degree + 1)
    step = np.pi / sectors
    samples = sectors // 2 * len(z)
    keep = (samples + 1) // 2 if transpose_symmetry else samples
    for polar, sample in product(range(len(mu)), range(keep)):
        m, j = divmod(sample, len(z))
        angle = step * (m + (1 + z[j]) / 2)
        multiplicity = 2 if transpose_symmetry and 2 * sample + 1 != samples else 1
        yield (mu[polar] * np.cos(angle), mu[polar] * np.sin(angle),
               multiplicity * polar_weights[polar] * w[j] / (2 * sectors))


def _sector_averages(sectors, polar_bands, transpose_symmetry, degree=0):
    """Yield five scalar coefficients and a weight for constant cells.

    Enriched cells carry the coefficients as a (mode, mode, 5) tensor and a weight.
    """
    polar_edges = np.cos(np.linspace(0.0, np.pi / 2, polar_bands + 1))
    polar_edges[-1] = 0.0
    polar_width = polar_edges[:-1] - polar_edges[1:]
    a, b = polar_edges[1:], polar_edges[:-1]
    mean_mu2 = 1 - (a * a + a * b + b * b) / 3
    primitive = (polar_edges * np.sqrt(1 - polar_edges**2) + np.arcsin(polar_edges)) / 2
    mean_mu = -np.diff(primitive) / polar_width

    step = np.pi / sectors
    samples = sectors // 2
    keep = (samples + 1) // 2 if transpose_symmetry else samples
    sinc = np.sinc(1 / sectors)
    half_sinc = np.sinc(0.5 / sectors)
    if degree:
        z, w = leggauss(12)
        modes = degree + 1
        basis = legvander(z, degree) * np.sqrt(2 * np.arange(modes) + 1)
        weights = np.einsum("ki,kj,k->kij", basis, basis, w / 2).reshape(z.size, -1)
    for polar, sample in product(range(polar_bands), range(keep)):
        phi = (sample + 0.5) * step
        square = mean_mu2[polar] / 2
        contrast = sinc * np.cos(2 * phi)
        multiplicity = 2 if transpose_symmetry and 2 * sample + 1 != samples else 1
        average = (square * (1 + contrast),
                   square * sinc * np.sin(2 * phi),
                   square * (1 - contrast),
                   mean_mu[polar] * half_sinc * np.cos(phi),
                   mean_mu[polar] * half_sinc * np.sin(phi))
        weight = multiplicity * polar_width[polar] / sectors
        if degree == 0:
            yield (*average, weight)
        else:
            cos_phi = np.cos(phi + step / 2 * z)
            sin_phi = np.sin(phi + step / 2 * z)
            coefficients = np.array([mean_mu2[polar] * cos_phi**2,
                                     mean_mu2[polar] * cos_phi * sin_phi,
                                     mean_mu2[polar] * sin_phi**2,
                                     mean_mu[polar] * cos_phi,
                                     mean_mu[polar] * sin_phi])
            # Legendre shapes have diagonal mass. Streaming and vacuum
            # retain the couplings between shapes within the finite sector.
            tensors = (coefficients @ weights).reshape(5, modes, modes).transpose(1, 2, 0)
            tensors[0, 0] = average
            yield tensors, weight


class _QuadraticMesh:
    def __init__(self, nx, ny, hx, hy, boundaries):
        grid = np.arange((2 * ny + 1) * (2 * nx + 1), dtype=np.int32)
        grid = grid.reshape(2 * ny + 1, 2 * nx + 1)
        self.cells = np.stack([
            grid[y:y + 2 * ny:2, x:x + 2 * nx:2].ravel()
            for y, x in product(range(3), repeat=2)
        ], axis=1)
        self.area = hx * hy
        self.aspect = hy / hx

        # Identify specular traces of the two fields. A single ID at a
        # reflecting corner also handles two intersecting mirror faces.
        edges = (grid[:, 0], grid[0], grid[:, -1], grid[-1])
        reflected = np.zeros(grid.size, dtype=bool)
        self.vacuum = []
        for side, (bc, edge) in enumerate(zip(boundaries, edges)):
            if bc == "reflect":
                reflected[edge] = True
            else:
                face_cells = (np.arange(0, nx * ny, nx), np.arange(nx),
                              np.arange(nx - 1, nx * ny, nx),
                              np.arange(nx * (ny - 1), nx * ny))[side]
                length = (hx if side % 2 else hy)[face_cells]
                face = np.stack((edge[:-2:2], edge[1:-1:2], edge[2::2]), axis=1)
                self.vacuum.append((side % 2, face, length))
        mirror = np.arange(grid.size, dtype=np.int32)
        mirror[~reflected] = grid.size + np.arange(np.count_nonzero(~reflected))
        self.mirror = mirror
        self.size = grid.size + np.count_nonzero(~reflected)
        self.grid_size = grid.size

    def prepare_bilinear(self):
        corners = self.cells[:, _CORNERS]
        ids = np.unique(np.stack((corners, self.mirror[corners])))
        self.bilinear_id = np.full(self.size, -1, dtype=np.int32)
        self.bilinear_id[ids] = np.arange(ids.size, dtype=np.int32)
        self.bilinear_size = ids.size

    def matrix(self, st, x, y):
        return self.tensor_matrix(st, x * x, x * y, y * y, x, y)

    def tensor_matrix(self, st, xx, xy, yy, vacuum_x, vacuum_y, *, mass_scale=1,
                      row_bilinear=False, col_bilinear=False):
        h = self.aspect
        mass_form, x_form, y_form, cross_form, face_form = _FORMS[row_bilinear, col_bilinear]
        mass = (mass_scale * st * self.area)[:, None, None] * mass_form
        x_term = (xx * h / st)[:, None, None] * x_form
        y_term = (yy / (h * st))[:, None, None] * y_form
        cross = (xy / st)[:, None, None] * cross_form
        diagonal = mass + x_term + y_term

        data, rows, cols = [], [], []

        def add(dofs, local, corners):
            row_ids = self.bilinear_id[dofs[:, corners]] if row_bilinear else dofs
            col_ids = self.bilinear_id[dofs[:, corners]] if col_bilinear else dofs
            rows.append(np.broadcast_to(row_ids[:, :, None], local.shape).ravel())
            cols.append(np.broadcast_to(col_ids[:, None, :], local.shape).ravel())
            data.append(local.ravel())

        add(self.cells, diagonal + cross, _CORNERS)
        add(self.mirror[self.cells], diagonal - cross, _CORNERS)
        for axis, face, length in self.vacuum:
            robin = ((vacuum_y if axis else vacuum_x) * length)[:, None, None] * face_form
            add(face, robin, [0, 2])
            add(self.mirror[face], robin, [0, 2])
        nrow = self.bilinear_size if row_bilinear else self.size
        ncol = self.bilinear_size if col_bilinear else self.size
        mat = coo_matrix((np.concatenate(data),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(nrow, ncol)).tocsc()
        mat.sort_indices()
        return mat

    def rhs(self, q):
        local = (q @ _MOMENTS) * self.area[:, None]
        first = np.bincount(self.cells.ravel(), local.ravel(), self.grid_size)
        second = np.bincount(self.mirror, first, self.size)
        second[:self.grid_size] += first
        return second

    def moments(self, field):
        both = field[:self.grid_size] + field[self.mirror]
        return both[self.cells] @ _MOMENTS.T


class SectorDiffusion(PSN2D):
    """Quadratic PSN sweep that uses the PSN-3 source-space Ritz solver."""

    def __init__(self, mat_map, h, st, scatter, nu_fission, *,
                 azimuth_degree=3, backend="off", M=12, I=30,
                 generic=False, **kwargs):
        if M < 2 or M % 2 or kwargs.get("full_2pi", False):
            raise ValueError("sector diffusion needs even M >= 2 on the half-plane")
        if (isinstance(azimuth_degree, bool) or not isinstance(azimuth_degree, int)
                or azimuth_degree < 0):
            raise ValueError("sector diffusion needs a nonnegative integer A")
        if generic and (I < 2 or I % 2):
            raise ValueError("sector diffusion needs even I >= 2 for the generic polar model")
        super().__init__(mat_map, h, st, scatter, nu_fission, M=M, I=I,
                         generic=generic, **kwargs)
        if len(self.bnd) != 4 or any(mode not in ("reflect", "vacuum")
                                     for mode in self.bnd):
            raise ValueError("sector diffusion supports reflect and vacuum boundaries")
        if np.any(~np.isfinite(self.St) | (self.St <= 0)):
            raise ValueError("sector diffusion needs positive finite total cross sections")
        if backend not in ("off", "chol-rs", "chol-packed-rs", "sector-cholmod"):
            raise ValueError("sector diffusion supports --opt off, chol-rs, chol-packed-rs or sector-cholmod")
        hx, hy = ((self.hx.ravel(), self.hy.ravel()) if self.rect else
                  (np.full(self.nodes, self.h), np.full(self.nodes, self.h)))
        if np.any(~np.isfinite(hx) | ~np.isfinite(hy) | (hx <= 0) | (hy <= 0)):
            raise ValueError("sector diffusion needs positive finite cell widths")
        self.mesh = _QuadraticMesh(self.nx, self.ny, hx, hy, self.bnd)
        # The fundamental mode is in the transpose-even subspace when the
        # physical model has that symmetry. Fold complementary azimuths there.
        self.transpose_symmetry = (
            self.nx == self.ny and self.bnd[::2] == self.bnd[1::2]
            and np.array_equal(self.mat, self.mat.T)
            and np.array_equal(hx.reshape(self.mat.shape), hy.reshape(self.mat.shape).T)
        )
        self.angles = list(self._angular_cells(azimuth_degree))
        self.backend = backend
        self._factors = []
        self._pool = None
        self._homes = []
        self.factor_bytes = 0

    def _angular_cells(self, azimuth_degree):
        if self.generic:
            z, w = leggauss(self.I)
            mu = np.sqrt(1 - ((1 + z) / 2) ** 2)
            polar_weights = w / 2
        else:
            mu, polar_weights = self.mu, self.W
        return _directions(self.M, azimuth_degree, mu, polar_weights,
                           self.transpose_symmetry)

    def _angular_matrix(self, st, cell):
        return self.mesh.matrix(st, *cell[:-1])

    def _angular_rhs(self, rhs):
        return rhs

    def _build_topology(self):
        self.nodes = self.nx * self.ny
        self.j_idx, self.i_idx = np.divmod(np.arange(self.nodes), self.nx)

    def _prefactor(self):
        matidx = self.mat.ravel()
        if self.backend == "sector-cholmod":
            from .sector_cholmod import SectorCholmod
            factor = SectorCholmod()
        elif self.backend == "off":
            def factor(matrix):
                lu = splu(matrix, permc_spec="MMD_AT_PLUS_A",
                          diag_pivot_thresh=0, options={"SymmetricMode": True})
                # Count both sparse triangular factors and their permutations.
                return lu.solve, lu.nnz * 12 + matrix.shape[0] * 16
        else:
            from .memopt import _load_rust_shared_chol
            RustChol = _load_rust_shared_chol().factor_cls
            if self.backend == "chol-packed-rs":
                from .packedchol import _PackedFactor, load_kernel
                kernel = load_kernel()

                def factor(matrix):
                    # Grazing Q2 systems can reach the FP64 residual floor
                    # of the more tightly scaled legacy face-current matrices.
                    chol = _PackedFactor(matrix.tocsr(), kernel,
                                         rtol=5e-12, maxiter=80)
                    return chol.solve, chol.pool_bytes
            else:
                def factor(matrix):
                    chol = RustChol(matrix)
                    return chol.solve, chol.diag()[4]

        jobs = [(g, cell) for g in range(self.ng) for cell in self.angles]
        self._factors = []
        self.factor_bytes = 0

        def build(job):
            g, cell = job
            matrix = self._angular_matrix(self.St[g, matidx], cell)
            solve, size = factor(matrix)
            return (g, cell[-1], solve), size

        started = perf_counter()
        worker_cap = int(os.environ.get("OPENPSN_PREFACTOR_WORKERS", "4"))
        if worker_cap < 1:
            raise ValueError("OPENPSN_PREFACTOR_WORKERS must be positive")
        workers = min(self.threads, worker_cap, len(jobs))
        self._homes = []
        if numa.mem_home() and numa.route_pools():
            allowed = os.sched_getaffinity(0)
            homes = [(node, cpus & allowed) for node, cpus in numa.node_cpus() if cpus & allowed]
            self._homes = homes if 1 < len(homes) <= workers else []
        pool = _PhasePool(workers, self._homes)
        try:
            # Bound the memory of transient matrices during setup. Every
            # angular system has the same grid graph and ordering.
            for first in range(0, len(jobs), workers):
                for entry, size in pool.map(build, jobs[first:first + workers], offset=first):
                    self._factors.append(entry)
                    self.factor_bytes += size
                    self._check_mem_gate()
                projected_gb = self.factor_bytes / len(self._factors) * len(jobs) / 2**30
                if projected_gb > self.mem_limit_gb:
                    raise MemoryError(
                        f"projected sector diffusion factor pool {projected_gb:.2f} GiB "
                        f"exceeds cap {self.mem_limit_gb:.2f} GiB")
        finally:
            pool.shutdown()
        self.factor_seconds = perf_counter() - started
        self._start_sweep_pool()

    def _check_mem_gate(self):
        if self.factor_bytes > self.mem_limit_gb * 2**30:
            raise MemoryError(
                f"sector diffusion factor pool {self.factor_bytes / 2**30:.2f} GiB "
                f"exceeds cap {self.mem_limit_gb:.2f} GiB")

    def _start_sweep_pool(self):
        if self._pool is None and self.threads > 1 and len(self._factors) > 1:
            self._pool = _PhasePool(min(self.threads, len(self._factors)), self._homes)

    def _stop_sweep_pool(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _transpose_average(self, moments):
        shape = (self.ng, self.ny, self.nx, 3, 3)
        view = moments.reshape(shape)
        return ((view + view.transpose(0, 2, 1, 4, 3)) * 0.5).reshape(moments.shape)

    def _sweep_all(self, q):
        if self.transpose_symmetry:
            q = self._transpose_average(q)
        rhs = [self._angular_rhs(self.mesh.rhs(group)) for group in q]

        def solve(block):
            g = block[0][0]
            field = np.zeros(self.mesh.size)
            for _, weight, apply in block:
                # The scalar flux picks out the mean of the angular cell.
                field += weight * apply(rhs[g])[:self.mesh.size]
            return g, field

        # Fixed contiguous blocks preserve the reduction order across thread
        # counts. Projection is linear, so do it once per energy group.
        block_size = 1 if self._homes else 4
        n = len(self.angles)
        blocks = [self._factors[g * n + i:g * n + min(i + block_size, n)]
                  for g in range(self.ng) for i in range(0, n, block_size)]
        fields = np.zeros((self.ng, self.mesh.size))
        execute = map if self._pool is None else self._pool.map
        for g, field in execute(solve, blocks):
            fields[g] += field
        moments = np.stack([self.mesh.moments(field) for field in fields])
        if self.transpose_symmetry:
            moments = self._transpose_average(moments)
        return moments[:, :, 0], moments

    def keff(self, max_outer=2000, outer_tol=1e-9, verbose=True):
        project = self._transpose_average if self.transpose_symmetry else None
        try:
            return keff_ritz(self, max_outer, outer_tol, verbose,
                             moment_count=_MOMENTS.shape[0], factor_threads=1,
                             project_source=project)
        finally:
            self._factors.clear()


class SectorAverage(SectorDiffusion):
    """Finite-angle-cell PSN with local azimuth shapes through degree two."""

    def __init__(self, *args, generic=True, azimuth_degree=0, slope_space="quadratic", **kwargs):
        if not generic or azimuth_degree not in (0, 1, 2):
            raise ValueError("sector_average requires angular.model=generic and A=0, 1 or 2")
        if slope_space not in ("quadratic", "bilinear"):
            raise ValueError("sector_average slope_space must be quadratic or bilinear")
        if slope_space == "bilinear" and not azimuth_degree:
            raise ValueError("bilinear slope_space requires A=1 or A=2")
        self.azimuth_degree = azimuth_degree
        self.slope_space = slope_space
        super().__init__(*args, generic=generic, azimuth_degree=azimuth_degree, **kwargs)
        if self.slope_space == "bilinear":
            self.mesh.prepare_bilinear()
        self.slope_size = self.mesh.bilinear_size if self.slope_space == "bilinear" else self.mesh.size

    def _angular_cells(self, azimuth_degree):
        return _sector_averages(self.M, self.I, self.transpose_symmetry, azimuth_degree)

    def _angular_matrix(self, st, cell):
        if not self.azimuth_degree:
            return self.mesh.tensor_matrix(st, *cell[:-1])
        tensors, _ = cell
        modes = self.azimuth_degree + 1
        blocks = [[None] * modes for _ in range(modes)]
        bilinear = self.slope_space == "bilinear"
        for row in range(modes):
            for col in range(row, modes):
                matrix = self.mesh.tensor_matrix(
                    st, *tensors[row, col], mass_scale=int(row == col),
                    row_bilinear=bilinear and row > 0, col_bilinear=bilinear and col > 0,
                )
                blocks[row][col] = matrix
                if row < col:
                    blocks[col][row] = matrix.T
        return bmat(blocks, format="csc")

    def _angular_rhs(self, rhs):
        if not self.azimuth_degree:
            return rhs
        # Isotropic sources project only onto the constant angular shape.
        source = np.zeros(rhs.size + self.azimuth_degree * self.slope_size)
        source[:rhs.size] = rhs
        return source
