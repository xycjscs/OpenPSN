"""Exact invariant-subspace assembly for mirror-symmetric phase meshes."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.sparse import bmat, coo_matrix

from .sector_diffusion import _CORNERS, _FORMS, _element_tensor


class ReflectedAssembly:
    def __init__(self, psn):
        if (tuple(psn.bnd) != ("reflect",) * 4
                or not np.array_equal(psn.mat, psn.mat[:, ::-1])
                or not np.array_equal(psn.mat, psn.mat[::-1, :])
                or (psn.rect and any(not np.array_equal(width, flip)
                                     for width in (psn.hx, psn.hy)
                                     for flip in (width[:, ::-1], width[::-1, :])))):
            raise ValueError("mirror reduction needs two exact assembly reflection symmetries")
        mesh = psn.mesh
        self.spatial_shape = psn.ny, psn.nx
        nodes = np.arange(mesh.grid_size).reshape(2 * psn.ny + 1, 2 * psn.nx + 1)
        mirrored = mesh.mirror[nodes]
        representative = np.empty(mesh.size, dtype=np.int32)
        # Each axial reflection interchanges the two signs of the mixed
        # derivative. Their composition preserves each sign field.
        representative[nodes] = np.minimum.reduce(
            (nodes, mirrored[:, ::-1], mirrored[::-1, :], nodes[::-1, ::-1]))
        representative[mirrored] = np.minimum.reduce(
            (mirrored, nodes[:, ::-1], nodes[::-1, :], mirrored[::-1, ::-1]))
        _, self.mean = np.unique(representative, return_inverse=True)
        self.mean_size = int(self.mean.max()) + 1
        self.cells = slice(0, (psn.nodes + 1) // 2)
        active = mesh.cells[self.cells]
        self.bilinear = tuple(psn._bilinear_mode(mode) for mode in range(psn.angular_modes))
        maps = {False: self.mean}
        nodes = {False: self.mean[active]}
        if any(self.bilinear):
            support = mesh.bilinear_id >= 0
            _, slope = np.unique(representative[support], return_inverse=True)
            slope_by_node = np.empty(mesh.size, dtype=np.int32)
            slope_by_node[support] = slope
            maps[True] = slope
            nodes[True] = slope_by_node[active[:, _CORNERS]]
        self.nodes = [nodes[flag] for flag in self.bilinear]
        self.shapes = [int(maps[flag].max()) + 1 for flag in self.bilinear]
        offsets = np.r_[0, np.cumsum(self.shapes)]
        self.indices = np.concatenate([maps[flag] + offset
                                       for flag, offset in zip(self.bilinear, offsets[:-1])])
        self.size = int(offsets[-1])
        self.weight = np.full(active.shape[0], 4.0)
        if psn.nodes % 2:
            self.weight[-1] = 2
        self.area = mesh.area[self.cells]
        self.aspect = mesh.aspect[self.cells]

    def matrix(self, matrix):
        entries = matrix.tocoo()
        return coo_matrix((entries.data, (self.indices[entries.row], self.indices[entries.col])),
                          shape=(self.size, self.size)).tocsr()

    def rhs(self, full):
        return np.bincount(self.indices, weights=full, minlength=self.size)

    def source(self, moments):
        view = moments.reshape(moments.shape[0], *self.spatial_shape, 3, 3)
        parity = np.array([1, -1, 1])
        reflected_x = (view + view[:, :, ::-1] * parity) * .5
        reflected_xy = (reflected_x + reflected_x[:, ::-1] * parity[:, None]) * .5
        return reflected_xy.reshape(moments.shape)

    def angular_matrix(self, st, cell):
        st = st[self.cells]
        modes = len(self.bilinear)
        tensors = np.asarray(cell[:-1]).reshape(1, 1, 5) if modes == 1 else cell[0]
        blocks = [[None] * modes for _ in range(modes)]
        # Reflection in one axis exchanges the sign fields. Central
        # reflection pairs the positive field's cells away from the centre.
        for row, col in combinations_with_replacement(range(modes), 2):
            forms = _FORMS[self.bilinear[row], self.bilinear[col]]
            xx, xy, yy, _, _ = tensors[row, col]
            diagonal, cross = _element_tensor(st, self.area, self.aspect, xx, xy, yy, forms, int(row == col))
            data = (diagonal + cross) * self.weight[:, None, None]
            rows = np.broadcast_to(self.nodes[row][:, :, None], data.shape)
            cols = np.broadcast_to(self.nodes[col][:, None, :], data.shape)
            block = coo_matrix((data.ravel(), (rows.ravel(), cols.ravel())),
                               shape=(self.shapes[row], self.shapes[col])).tocsc()
            blocks[row][col] = block
            if col != row:
                blocks[col][row] = block.T
        return bmat(blocks, format="csc")
