"""Diffusion-preconditioned projection for the PSN transport eigenproblem."""
from time import perf_counter

import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import LinearOperator, eigs, splu

from . import runtime
from .cmfd import Cmfd

_SPACE_SIZE = 32
_RETAIN = 5
_COARSE_BLOCK = 2
# Keep the shifted diffusion correction away from its fundamental pole.
_SHIFT_RATIO = 1.125
_SOURCE_TOL_FACTOR = 5.0


def _vacuum_leakage(psn, mesh):
    """Integrated diffusion leakage at vacuum faces (zero at reflectors)."""
    leak = np.zeros_like(mesh.D)
    axes = (psn.i_idx, psn.j_idx, psn.i_idx, psn.j_idx)
    edges = (0, 0, psn.nx - 1, psn.ny - 1)
    for side, (bc, index, edge) in enumerate(zip(psn.bnd, axes, edges)):
        if bc != "vacuum":
            continue
        mask = index == edge
        cells = mesh.cid[mask]
        normal = psn.h
        if psn.rect:
            widths = psn.hy if side % 2 else psn.hx
            normal = widths.ravel()[mask]
        length = mesh.V[mask] / normal
        d = mesh.D[:, cells]
        conductance = d * length / (0.5 * normal + 2.0 * d)
        for g, values in enumerate(conductance):
            leak[g] += np.bincount(cells, weights=values, minlength=mesh.nC)
    return leak


def _coarse_setup(psn, shape):
    """Build a starting source and a fixed preconditioner on the same mesh."""
    mesh = Cmfd(psn, block=_COARSE_BLOCK)
    ng, nc = mesh.ng, mesh.nC
    mesh._tally(np.ones(shape[:2]))
    loss, fission = mesh._operators(
        transport_split=True, boundary_leak=_vacuum_leakage(psn, mesh))
    fission.eliminate_zeros()
    loss.eliminate_zeros()
    solve_loss = splu(loss, permc_spec="MMD_AT_PLUS_A").solve

    # The emissions in every coarse cell share chi. Project the initial
    # diffusion eigenproblem to one fission rate per cell.
    chi = psn.chi[:, None]
    production = mesh.Fis.sum(axis=0) / chi.sum()

    def flux_from_fission(source):
        return solve_loss((chi * source).ravel()).reshape(ng, nc)

    def fission_response(source):
        return np.sum(production * flux_from_fission(source), axis=0)

    if nc <= 2:
        matrix = np.column_stack([fission_response(v) for v in np.eye(nc)])
        values, vectors = eig(matrix)
        dominant = np.argmax(values.real)
        k, mode = values[dominant], vectors[:, dominant]
    else:
        operator = LinearOperator((nc, nc), fission_response, dtype=float)
        values, vectors = eigs(operator, k=1, which="LR", v0=np.ones(nc),
                               tol=1e-6)
        k, mode = values[0], vectors[:, 0]
    if not np.isfinite(k) or k.real <= 0 or abs(k.imag) > 1e-10 * k.real:
        raise RuntimeError("Ritz initialization found no positive diffusion eigenvalue")
    k = float(k.real)
    mode = mode.real
    if mode.sum() < 0:
        mode = -mode

    def coarse_source(coupling, flux):
        return np.einsum("ghc,hc->gc", coupling, flux) / mesh.VC

    flux = flux_from_fission(mode)
    initial = np.zeros(shape)
    initial[:, :, 0] = coarse_source(mesh.Sca + mesh.Fis / k, flux)[:, mesh.cid]
    del solve_loss

    # If T is the diffusion loss before collisions and U=S+F/shift,
    # the source-space inverse is (I-U T^-1)^-1 = I+U(T-U)^-1.
    # Apply the correction to constant moments only; fine sweeps still
    # evaluate all spatial moments and determine the eigenpair.
    shift = _SHIFT_RATIO * k
    solve_shifted = splu(loss - fission / shift, permc_spec="MMD_AT_PLUS_A").solve
    coupling = mesh.Sca + mesh.Fis / shift

    def precondition(residual):
        direction = residual.reshape(shape).copy()
        rhs = np.array([
            np.bincount(mesh.cid, weights=mesh.V * row, minlength=nc)
            for row in direction[:, :, 0]
        ])
        flux = solve_shifted(rhs.ravel()).reshape(ng, nc)
        direction[:, :, 0] += coarse_source(coupling, flux)[:, mesh.cid]
        return direction.ravel()

    return initial, precondition


class _RitzSpace:
    """Orthonormal source basis plus cached left and right transport images."""

    def __init__(self, depth, size):
        self.basis = np.empty((depth, size))
        self.left = np.empty_like(self.basis)
        self.right = np.empty_like(self.basis)
        self.loss = np.empty((depth, depth))
        self.production = np.empty_like(self.loss)
        self.rates = np.empty(depth)
        self.count = 0

    def unit_direction(self, direction):
        basis = self.basis[:self.count]
        # Reorthogonalize before the sweep so each cached image corresponds
        # to the final basis vector, including near convergence.
        norm_before = np.linalg.norm(direction)
        for _ in range(2):
            direction = direction - (basis @ direction) @ basis
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm <= np.finfo(float).eps * norm_before:
            raise RuntimeError("Ritz search space exhausted before convergence")
        return direction / norm

    def append(self, direction, loss, production, rate):
        j = self.count
        self.count += 1
        self.basis[j], self.left[j], self.right[j] = direction, loss, production
        self.rates[j] = rate
        basis = self.basis[:self.count]
        self.loss[:self.count, j] = basis @ loss
        self.loss[j, :j] = self.left[:j] @ direction
        self.production[:self.count, j] = basis @ production
        self.production[j, :j] = self.right[:j] @ direction

    def eigenvectors(self):
        n = self.count
        values, vectors = eig(self.production[:n, :n], self.loss[:n, :n])
        select = np.flatnonzero(np.isfinite(values) & (values.real > 0.0)
                                & (np.abs(values.imag) <= 1e-9 * values.real))
        order = select[np.argsort(values[select].real)[::-1]]
        return values[order].real, vectors[:, order].real

    def restart(self, vectors):
        """Retain a small invariant subspace without repeating its sweeps."""
        if not vectors.shape[1]:
            self.count = 0
            return
        n = self.count
        keep = min(_RETAIN, n - 1)
        u, sigma, _ = np.linalg.svd(vectors[:, :keep], full_matrices=False)
        self.count = int(np.count_nonzero(sigma > sigma[0] * 1e-12))
        transform = u[:, :self.count].T
        m = self.count
        self.rates[:m] = transform @ self.rates[:n]
        for array in (self.basis, self.left, self.right):
            array[:m] = transform @ array[:n]
        self.loss[:m, :m] = self.basis[:m] @ self.left[:m].T
        self.production[:m, :m] = self.basis[:m] @ self.right[:m].T


def keff_ritz(psn, max_sweeps, outer_tol, verbose, *, moment_count=5,
              factor_threads=None, project_source=None, initial_source=None):
    """Solve Bq = kAq using exact sweeps for A=I-S*T and B=F*T.

    T maps a source to its transport flux moments. The coarse diffusion
    operator proposes search directions and a default starting source.
    A supplied starting source uses the same group, cell and moment shape.
    A source projector may restrict the search to an invariant subspace.
    A fresh full sweep certifies the final source residual and fission balance.
    """
    if max_sweeps < 2 or not np.isfinite(outer_tol) or outer_tol <= 0.0:
        raise ValueError("Ritz requires at least two sweeps and a positive tolerance")
    if any(mode not in ("vacuum", "reflect") for mode in psn.bnd):
        raise ValueError("Ritz supports vacuum and reflective boundaries")
    matidx = psn.mat[psn.j_idx, psn.i_idx]
    nu = psn.nuSf[:, matidx]
    rate_weights = nu * psn.area.ravel() if psn.rect else nu
    reference_rate = float(rate_weights.sum())
    if reference_rate <= 0 or psn.chi.sum() <= 0:
        raise ValueError("Ritz requires positive fission production")
    shape = (psn.ng, psn.nodes, moment_count)
    if initial_source is not None:
        initial_source = np.asarray(initial_source, dtype=float)
        if initial_source.shape != shape:
            raise ValueError(f"initial_source must be a nonzero finite array of shape {shape}")
        if project_source is not None:
            initial_source = project_source(initial_source)
        if not np.isfinite(initial_source).all() or not np.any(initial_source):
            raise ValueError("initial_source must remain finite and nonzero after projection")
    previous_threads = runtime.get_thread_limit()
    sweeps = 0
    verification_sweeps = 0
    sweep_times = []
    last_k = None
    residual = np.inf
    try:
        runtime.set_thread_limit(factor_threads or psn.threads)
        psn._prefactor()
        psn._check_mem_gate()
        runtime.set_thread_limit(1)
        initial, precondition = _coarse_setup(psn, shape)
        packed = getattr(psn, "_memopt_backend", None) == "chol-packed-rs"
        if packed:
            psn._start_sweep_pool()
        if initial_source is not None:
            initial = initial_source
        elif project_source is not None:
            initial = project_source(initial)
        direction = initial.ravel()
        depth = min(_SPACE_SIZE, direction.size)
        space = _RitzSpace(depth, direction.size)
        source_tol = _SOURCE_TOL_FACTOR * outer_tol
        while sweeps < max_sweeps:
            q = space.unit_direction(direction)
            started = perf_counter()
            phi, moments = psn._sweep_all(q.reshape(shape))
            sweep_times.append(perf_counter() - started)
            if sweeps == 0 and not packed:
                psn._start_sweep_pool()
            sweeps += 1
            loss = q - psn._source_update(moments, np.inf, matidx).ravel()
            fission = np.einsum("gn,gnm->nm", nu, moments)
            production = (psn.chi[:, None, None] * fission).ravel()
            rate = float(np.sum(rate_weights * phi))
            space.append(q, loss, production, rate)
            values, vectors = space.eigenvectors()
            if values.size:
                k, weights = float(values[0]), vectors[:, 0]
                n = space.count
                balance = (weights @ space.right[:n]) / k - weights @ space.left[:n]
                residual = float(np.linalg.norm(balance))
                change = abs(k - last_k) / k if last_k is not None else np.inf
                if verbose and (sweeps == 1 or sweeps % 5 == 0):
                    print(f"  ritz sweep {sweeps}: keff={k:.10f}, source={residual:.2e}")
                if residual <= source_tol and sweeps < max_sweeps:
                    emission = weights @ space.rates[:n]
                    if emission != 0.0 and (change <= outer_tol or n == 1):
                        candidate = (weights @ space.basis[:n]).reshape(shape)
                        candidate *= k * reference_rate / emission
                        started = perf_counter()
                        phi, moments = psn._sweep_all(candidate)
                        sweep_times.append(perf_counter() - started)
                        sweeps += 1
                        verification_sweeps += 1
                        k_verified = float(np.sum(rate_weights * phi) / reference_rate)
                        k_error = abs(k_verified - k) / k
                        if k_verified > 0.0 and k_error <= outer_tol:
                            updated = psn._source_update(moments, k_verified, matidx)
                            residual = (np.linalg.norm(updated - candidate)
                                        / np.linalg.norm(candidate))
                            if residual <= source_tol:
                                psn.outer_iterations = sweeps - verification_sweeps
                                psn.extra_sweeps = verification_sweeps
                                psn.sweeps_executed = sweeps
                                psn.ritz_sweep_times = sweep_times
                                psn.source_residual = float(residual)
                                psn.keff_residual = max(k_error, change if n > 1 else 0.0)
                                return k_verified, phi, candidate
                last_k = k
            else:
                balance = production - loss
                last_k = None
            direction = precondition(balance)
            if project_source is not None:
                direction = project_source(direction.reshape(shape)).ravel()
            if space.count == depth:
                space.restart(vectors)
    finally:
        psn._stop_sweep_pool()
        runtime.set_thread_limit(previous_threads)
    raise RuntimeError(f"Ritz did not converge in {sweeps} transport sweeps "
                       f"(last relative source residual {residual:.3e})")
