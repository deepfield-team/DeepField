"""Classes and routines for handling model grids."""
import warnings

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from .decorators import cached_property, apply_to_each_input
from .base_spatial import SpatialComponent
from .grid_utils import get_top_z_coords, numba_get_volumes, numba_get_xyz, get_connectivity_matrix
from .utils import rolling_window, mk_vtk_id_list, get_single_path
from .parse_utils import read_ecl_bin


class Grid(SpatialComponent):
    """Basic grid class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vtk_grid = None
        self._cell_id_d = None
        self._vtk_grid_params = {'use_only_active': True, 'cell_size': None, 'scaling': True}
        self._vtk_grid_scales = [1, 1, 1]
        self._vtk_locator = None
        self.to_spatial()
        if 'MAPAXES' not in self:
            setattr(self, 'MAPAXES', np.array([0, 1, 0, 0, 1, 0]))
        if 'ACTNUM' not in self and 'DIMENS' in self:
            self.actnum = np.ones(self.dimens, dtype=bool)

    @property
    def origin(self):
        """Grid axes origin relative to the map coordinates."""
        return np.array([self.mapaxes[2], self.mapaxes[3], self.tops.ravel()[0]])

    @property
    def xyz(self):
        """Cell vertices coordinates."""
        raise NotImplementedError()

    @property
    def cell_centroids(self):
        """Centroids of cells."""
        return self.xyz.mean(axis=-2)

    @property
    def cell_volumes(self):
        """Volumes of cells."""
        grid = specify_grid(self)
        return grid.cell_volumes

    def to_corner_point(self):
        """Corner-point representation of the grid."""
        raise NotImplementedError()

    @property
    def as_corner_point(self):
        """Corner-point representation of the grid."""
        raise NotImplementedError()

    @cached_property
    def bounding_box(self):
        """Pair of diagonal corner points for grid's bounding box."""
        min_vals = self.xyz.min(axis=-2).min(axis=(0, 1, 2))
        max_vals = self.xyz.max(axis=-2).max(axis=(0, 1, 2))
        return np.array([min_vals, max_vals])

    @property
    def ex(self):
        """Unit vector along grid X axis."""
        ex = np.array([self.mapaxes[-2] - self.mapaxes[2],
                       self.mapaxes[-1] - self.mapaxes[3]])
        return ex / np.linalg.norm(ex)

    @property
    def ey(self):
        """Unit vector along grid Y axis."""
        ey = np.array([self.mapaxes[0] - self.mapaxes[2],
                       self.mapaxes[1] - self.mapaxes[3]])
        return ey / np.linalg.norm(ey)

    def _load_ecl_binary(self, path_to_results, attrs, basename, logger=None,
                         **kwargs):
        _ = kwargs
        path = get_single_path(path_to_results, basename + '.EGRID', logger)
        if path is None:
            return
        attrs_tmp = attrs + ['GRIDHEAD'] if 'DIMENS' in attrs else attrs
        sections = read_ecl_bin(path, attrs_tmp, logger=logger)
        if 'DIMENS' in attrs:
            setattr(self, 'DIMENS', sections['GRIDHEAD'][1:4])
        for k in ['ZCORN', 'COORD', 'MAPAXES', 'ACTNUM']:
            if (k in attrs) and (k in sections):
                if k == 'ACTNUM':
                    setattr(self, 'ACTNUM', sections['ACTNUM'].astype(bool))
                else:
                    setattr(self, k, sections[k])
                self.state.binary_attributes.append(k)

    def _read_buffer(self, buffer, attr, logger=None, **kwargs):
        if attr == 'DIMENS':
            super()._read_buffer(buffer, attr, dtype=int, logger=logger, compressed=False)
        elif attr in ['DX', 'DY', 'DZ', 'TOPS']:
            super()._read_buffer(buffer, attr, dtype=float, logger=logger, compressed=True)
        elif attr in ['DXV', 'DYV', 'DZV']:
            super()._read_buffer(buffer, attr[:2], dtype=float, logger=logger, compressed=True)
            data = getattr(self, attr[:2])
            data = (data.reshape(-1, 1, 1) if attr == 'DXV' else
                    data.reshape(1, -1, 1) if attr == 'DYV' else
                    data.reshape(1, 1, -1))
            setattr(self, attr[:2], (np.zeros(self.dimens) + data).ravel(order='F'))
            logger.info("Keyword {} was converted to {}.".format(attr, attr[:2]))
        elif attr == 'ZCORN':
            super()._read_buffer(buffer, attr, dtype=float, compressed=True)
        elif attr == 'COORD':
            super()._read_buffer(buffer, attr, dtype=float, compressed=True)
        elif attr == 'MINPV':
            super()._read_buffer(buffer, attr, dtype=float, compressed=False)
        elif attr in 'ACTNUM':
            super()._read_buffer(buffer, attr, dtype=lambda x: bool(int(x)), logger=logger, compressed=True)
        else:
            super()._read_buffer(buffer, attr, logger=logger, **kwargs)

    def apply_minpv(self):
        """Apply MINPV threshold to ACTNUM."""
        minpv_value = self.minpv[0]
        volumes = self.cell_volumes
        poro = self.field.rock.poro
        ntg = getattr(self.field.rock, "ntg", 1)
        mask = poro * volumes*ntg >= minpv_value
        self.actnum = self.actnum * mask

    @apply_to_each_input
    def _to_spatial(self, attr, **kwargs):
        """Spatial order 'F' transformations."""
        _ = kwargs
        data = getattr(self, attr)
        if isinstance(data, np.ndarray) and data.ndim == 1:
            if attr in ['ACTNUM', 'DX', 'DY', 'DZ', 'TOPS']:
                data = data.reshape(self.dimens, order='F')
            elif attr == 'COORD':
                nx, ny, nz = self.dimens
                data = data.reshape(-1, 6)
                data = data.reshape((nx + 1, ny + 1, 6), order='F')
            elif attr == 'ZCORN':
                nx, ny, nz = self.dimens
                data = data.reshape((2, nx, 2, ny, 2, nz), order='F')
                data = np.moveaxis(data, range(6), (3, 0, 4, 1, 5, 2))
                data = data.reshape((nx, ny, nz, 8), order='F')
            else:
                return self
            setattr(self, attr, data)
        return self

    @apply_to_each_input
    def _ravel(self, attr, **kwargs):
        """Ravel order 'F' transformations."""
        _ = kwargs
        data = getattr(self, attr)
        if attr in ['ACTNUM', 'DX', 'DY', 'DZ', 'TOPS']:
            data = data.ravel(order='F')
        elif attr == 'COORD':
            data = data.reshape((-1, 6), order='F').ravel()
        elif attr == 'ZCORN':
            nx, ny, nz = self.dimens
            data = data.reshape((nx, ny, nz, 2, 2, 2), order='F')
            data = np.moveaxis(data, (3, 0, 4, 1, 5, 2), range(6)).ravel(order='F')
        else:
            data = super()._ravel(attr=attr, order='F')
        return data

    def _make_data_dump(self, attr, fmt=None, float_dtype=None, **kwargs):
        """Prepare data for dump."""
        if fmt.upper() != 'HDF5':
            return super()._make_data_dump(attr, fmt=fmt, **kwargs)
        data = self.ravel(attr=attr)
        if attr == 'ACTNUM':
            return data.astype(bool)
        if attr in ['ZCORN', 'COORD', 'DX', 'DY', 'DZ', 'TOPS', 'MAPAXES']:
            return data if float_dtype is None else data.astype(float_dtype)
        if attr == 'DIMENS':
            return data.astype(int)
        return data

    def minimal_active_slices(self):
        """Get minimal cube slice that contains all active cells."""
        pos = np.where(self.actnum)
        return tuple(slice(p.min(), p.max() + 1) for p in pos)

    def crop_minimal_cube(self):
        """Crop to minimal cube containing active cells."""
        raise NotImplementedError()

    def get_neighbors_matrix(self, connectivity=1, fill_value=-1, ravel_index=False):
        """Get indices of neighbors for all active cells.

        Parameters
        ----------
        connectivity : int, optional
            Maximum number of orthogonal hops to consider a cell is
            as a neighbor, by default 1.
        fill_value : int, optional
            Value to fill indices of inactive or absent neighbors, by default -1.
        ravel_index : bool, optional
            Indices in raveled form, by default False.

        Returns
        -------
        res : misc
            Matrix of active neighbors and matrix of distances if 'calculate_distances'.
        """
        actnum = self.actnum
        res, invalid_cells = get_connectivity_matrix(actnum, connectivity)
        if ravel_index:
            res = np.ravel_multi_index((res[..., 0], res[..., 1], res[..., 2]), self.dimens, order='F')
        res[invalid_cells] = fill_value
        return res

    def calculate_neighbours_distances(self, connectivity=1, fill_value=-1, neighbours_matrix=None):
        """Calculate distances between neighbors for all active cells.

        Parameters
        ----------
        connectivity : int, optional
            Maximum number of orthogonal hops to consider a cell is
            as a neighbor., by default 1.
        fill_value : int, optional
            Value to fill indices of inactive or absent neighbors, by default -1.
        neighbours_matrix: numpy.ndarray, optional
            Matrix with indices of neighbors in unraveled form.

        Returns
        -------
        numpy.ndarray
            Matrix of distances.
        """
        actnum = self.actnum
        if neighbours_matrix is None:
            neighbours_matrix = self.get_neighbors_matrix(
                connectivity=connectivity,
                fill_value=fill_value,
                ravel_index=False
            )
        invalid_cells = np.where(
            np.any(neighbours_matrix == fill_value, axis=-1),
            np.ones(shape=neighbours_matrix.shape[:2]),
            np.zeros(shape=neighbours_matrix.shape[:2])
        ).astype(bool)
        neighbor_centers = self.cell_centroids[
            neighbours_matrix[..., 0], neighbours_matrix[..., 1], neighbours_matrix[..., 2]
        ]
        active_cells_centers = self.cell_centroids[actnum]
        distances = np.linalg.norm(
            neighbor_centers - np.tile(active_cells_centers[:, np.newaxis, :],
                                       (1, neighbor_centers.shape[1], 1)),
            axis=2
        )
        distances[invalid_cells] = fill_value
        return distances

    def create_vtk_grid(self, use_only_active=True, scaling=True, **kwargs):
        """Creates pyvista unstructured grid object.

        Returns
        -------
        grid : `pyvista.core.pointset.UnstructuredGrid` object
        """
        _ = kwargs
        self._vtk_grid_params = {'use_only_active': use_only_active, 'cell_size': None, 'scaling': scaling}

        cells = self.xyz
        indexes = np.moveaxis(np.indices(self.dimens), 0, -1)

        if use_only_active:
            cells = cells[self.actnum]
            indexes = indexes[self.actnum]
        else:
            cells = cells.reshape((-1,) + cells.shape[3:])
            indexes = indexes.reshape((-1,) + indexes.shape[3:])

        self._cell_id_d = dict(enumerate(indexes))

        cells[:, [2, 3]] = cells[:, [3, 2]]
        cells[:, [6, 7]] = cells[:, [7, 6]]

        n_cells = cells.shape[0]
        cells = cells.reshape(-1, cells.shape[-1], order='C')

        cells = numpy_support.numpy_to_vtk(cells, deep=True)

        points = vtk.vtkPoints()
        points.SetNumberOfPoints(8*n_cells)
        points.SetData(cells)

        cell_array = vtk.vtkCellArray()
        cell = vtk.vtkHexahedron()

        connectivity = np.insert(range(8 * n_cells), range(0, 8 * n_cells, 8), 8).astype(np.int64)
        cell_array.SetCells(n_cells, numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True))

        self._vtk_grid = vtk.vtkUnstructuredGrid()
        self._vtk_grid.SetPoints(points)
        self._vtk_grid.SetCells(cell.GetCellType(), cell_array)


class OrthogonalGrid(Grid):
    """Orthogonal uniform grid."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'TOPS' not in self and 'DZ' in self:
            tops = np.zeros(self.dimens)
            tops[..., 1:] = np.cumsum(self.dz, axis=-1)[..., :-1]
            setattr(self, 'TOPS', tops)

    @property
    def cell_volumes(self):
        """Volumes of cells."""
        return self.dx * self.dy * self.dz

    @property
    def xyz(self):
        """Cells' vertices coordinates."""
        xyz = np.zeros(tuple(self.dimens) + (8, 3))
        px = np.cumsum(self.dx, axis=0) + self.origin[0]
        py = np.cumsum(self.dy, axis=1) + self.origin[1]
        xyz[1:, :, :, [0, 2, 4, 6], 0] = px[:-1, :, :, None]
        xyz[:, :, :, [1, 3, 5, 7], 0] = px[..., None]
        xyz[:, 1:, :, [0, 1, 4, 5], 1] = py[:, :-1, :, None]
        xyz[:, :, :, [2, 3, 6, 7], 1] = py[..., None]
        xyz[:, :, :, :4, 2] = self.tops[..., None]
        xyz[:, :, :, 4:, 2] = (self.tops + self.dz)[..., None]
        return xyz

    def cell_sizes(self, cell_indices):
        """Cell sizes."""
        return np.array([self.dx[cell_indices],
                         self.dy[cell_indices],
                         self.dz[cell_indices]])

    def crop_minimal_cube(self):
        """Crop to minimal cube containing active cells.

        Returns
        -------
        grid : OrthoghonalGrid
            New grid.
        """
        min_slices = self.minimal_active_slices()
        actnum = self.actnum[min_slices]
        grid = self.__class__(
            actnum=actnum,
            dx=self.dx[min_slices],
            dy=self.dy[min_slices],
            dz=self.dz[min_slices],
            tops=self.tops[min_slices],
            dimens=actnum.shape)
        return grid

    def upscale(self, factors=(2, 2, 2), actnum_upscale='vote'):
        """Merge grid cells according to factors given.

        Parameters
        ----------
        factors : tuple, int
            Scale factors along each axis. If int, factors are the same for each axis.
        actnum_upscale : str
            Method to actnum upscaling. If 'vote', upscaled cell is active if majority
            of finer cells are active. If 'any', upscaled cell is active if any
            of finer cells is active. Default to 'vote'.

        Returns
        -------
        grid : OrthogonalGrid
            Merged grid.
        """
        factors = np.atleast_1d(factors)
        if factors.size == 1:
            factors = np.repeat(factors, 3)

        dx = np.sum(rolling_window(self.dx, factors), axis=(-3, -2, -1)) / (factors[1] + factors[2])
        dy = np.sum(rolling_window(self.dy, factors), axis=(-3, -2, -1)) / (factors[0] + factors[2])
        dz = np.sum(rolling_window(self.dz, factors), axis=(-3, -2, -1)) / (factors[0] + factors[1])
        tops = rolling_window(self.tops, factors)[..., 0].mean(axis=(-2, -1))

        out = rolling_window(self.actnum, factors)
        if actnum_upscale == 'vote':
            actnum = np.mean(out, axis=(-3, -2, -1)) > 0.5
        elif actnum_upscale == 'any':
            actnum = np.sum(out, axis=(-3, -2, -1)) > 0
        else:
            raise ValueError('Unknown mode of actnum upscaling: {}.'
                             .format(actnum_upscale))

        grid = self.__class__(dimens=actnum.shape, dx=dx, dy=dy, dz=dz,
                              actnum=actnum, tops=tops, mapaxes=self.mapaxes)

        return grid

    def downscale(self, factors=(2, 2, 2)):
        """Split grid cells according to factors given.

        Parameters
        ----------
        factors : tuple, int
            Scale factors along each axis. If int, factors are the same for each axis.

        Returns
        -------
        grid : OrthogonalGrid
            Refined grid.
        """
        factors = np.atleast_1d(factors)
        if factors.size == 1:
            factors = np.repeat(factors, 3)
        dimens = self.dimens * factors

        dx = np.kron(self.dx/factors[0], np.ones(factors))
        dy = np.kron(self.dy/factors[1], np.ones(factors))
        dz = np.kron(self.dz/factors[2], np.ones(factors))
        tops = np.kron(self.tops, np.ones(factors))
        for i in range(1,factors[2]):
            tops[:, :, i::factors[2]] += i*dz[:, :, :-i:factors[2]]

        actnum = np.kron(self.actnum, np.ones(factors))

        grid = self.__class__(dimens=dimens, dx=dx, dy=dy, dz=dz,
                              actnum=actnum, tops=tops, mapaxes=self.mapaxes)

        return grid

    def to_corner_point(self):
        """Create corner point representation of the current grid.

        Returns
        -------
        grid : CornerPointGrid
        """
        nx, ny, nz = self.dimens
        x0, y0, z0 = self.origin

        dx = self.dx[:, :1, :1]
        if (abs(self.dx - dx) > 0).any():
            raise ValueError('Can not convert irregular DX to corner point.')
        px = np.cumsum(np.hstack(([0], dx.ravel())))

        dy = self.dy[:1, :, :1]
        if (abs(self.dy - dy) > 0).any():
            raise ValueError('Can not convert irregular DY to corner point.')
        py = np.cumsum(np.hstack(([0], dy.ravel())))

        x_y = np.vstack([np.tile(px, len(py)), np.repeat(py, len(px))]).T
        x_y[:, 0] += x0
        x_y[:, 1] += y0

        coord = np.hstack((x_y,
                           np.ones(((ny + 1) * (nx + 1), 1)) * z0,
                           x_y,
                           np.ones(((ny + 1) * (nx + 1), 1)) * (z0 + nz))).ravel()

        zcorn = np.hstack([np.repeat(self.tops.ravel(order='F'), 4).reshape(nz, -1),
                           np.repeat(self.tops.ravel(order='F') +
                                     self.dz.ravel(order='F'), 4).reshape(nz, -1)]).reshape(2*nz, -1)
        zcorn = zcorn.ravel()

        grid = CornerPointGrid(dimens=self.dimens, mapaxes=self.mapaxes, actnum=self.actnum,
                               zcorn=zcorn.astype(float), coord=coord.astype(float))

        return grid

    @cached_property
    def _as_corner_point(self):
        """Cached CornerPoint representation of the current grid."""
        return self.to_corner_point()

    @property
    def as_corner_point(self):
        """Creates CornerPoint representation of the current grid."""
        return self._as_corner_point


class CornerPointGrid(Grid):
    """Corner point grid."""

    @property
    def origin(self):
        """Grid axes origin relative to the map coordinates."""
        return np.array([self.mapaxes[2], self.mapaxes[3], self.zcorn[0, 0, 0, 0]])

    @cached_property
    def _xyz(self):
        """Cached xyz property with x, y, z coordinates of cells vertices."""
        return numba_get_xyz(self.dimens, self.zcorn, self.coord)

    @property
    def xyz(self):
        """x, y, z coordinates of cells vertices."""
        return self._xyz

    @property
    def cell_volumes(self):
        """Volumes of cells."""
        return numba_get_volumes(self.xyz)

    def minimal_active_bounds(self):
        """Get z coordinates of top and bottom bounds of active cells."""
        zcorn = self.zcorn.reshape(self.zcorn.shape[:3] + (2, 4))
        z_top = get_top_z_coords(zcorn, self.actnum)
        z_bottom = -get_top_z_coords(-zcorn[:, :, ::-1, ::-1], self.actnum[:, :, ::-1])
        return z_top, z_bottom

    def upscale(self, factors=(2, 2, 2), actnum_upscale='vote'):
        """Upscale grid according to factors given.

        Parameters
        ----------
        factors : tuple, int
            Scale factors along each axis. If int, factors are the same for each axis.
        actnum_upscale : str
            Method to actnum upscaling. If 'vote', upscaled cell is active if majority
            of finer cells are active. If 'any', upscaled cell is active if any
            of finer cells is active. Default to 'vote'.

        Returns
        -------
        grid : CornerPointGrid
            Upscaled grid.
        """
        factors = np.atleast_1d(factors)
        if factors.size == 1:
            factors = np.repeat(factors, 3)
        coord = self.coord[::factors[0], ::factors[1]]
        dimens = self.dimens // factors
        d0, d1, d2 = dimens * factors
        s0, s1, s2 = factors
        zcorn = np.zeros(tuple(dimens) + (8,), dtype=self.zcorn.dtype)
        zcorn[:, :, :, 0] = self.zcorn[:d0:s0, :d1:s1, :d2:s2, 0]
        zcorn[:, :, :, 1] = self.zcorn[s0 - 1:d0:s0, :d1:s1, :d2:s2, 1]
        zcorn[:, :, :, 2] = self.zcorn[:d0:s0, s1 - 1:d1:s1, :d2:s2, 2]
        zcorn[:, :, :, 3] = self.zcorn[s0 - 1:d0:s0, s1 - 1:d1:s1, :d2:s2, 3]
        zcorn[:, :, :, 4] = self.zcorn[:d0:s0, :d1:s1, s2 - 1:d2:s2, 4]
        zcorn[:, :, :, 5] = self.zcorn[s0 - 1:d0:s0, :d1:s1, s2 - 1:d2:s2, 5]
        zcorn[:, :, :, 6] = self.zcorn[:d0:s0, s1 - 1:d1:s1, s2 - 1:d2:s2, 6]
        zcorn[:, :, :, 7] = self.zcorn[s0 - 1:d0:s0, s1 - 1:d1:s1, s2 - 1:d2:s2, 7]

        out = rolling_window(self.actnum, factors)
        if actnum_upscale == 'vote':
            actnum = np.mean(out, axis=(-3, -2, -1)) > 0.5
        elif actnum_upscale == 'any':
            actnum = np.sum(out, axis=(-3, -2, -1)) > 0
        else:
            raise ValueError('Unknown mode of actnum upscaling: {}.'.format(actnum_upscale))

        grid = self.__class__(coord=coord, dimens=dimens, zcorn=zcorn,
                              mapaxes=self.mapaxes, actnum=actnum)
        return grid

    def orthogonalize(self, dimens, only_active=False):
        """Construct orthogonal grid. Actnum tranfer should be made separately.

        Parameters
        ----------
        dimens : tuple, int
            Dimensions of the output orthogonal grid.
        only_active : bool
            Limits the new grid to active cells.

        Returns
        -------
        grid : OrthogonalGrid
            Orthogonal grid.
        """
        nx, ny, nz = dimens
        xyz_corn = self.xyz[self.actnum] if only_active else self.xyz
        x_min = xyz_corn[..., 0].min()
        x_max = xyz_corn[..., 0].max()
        y_min = xyz_corn[..., 1].min()
        y_max = xyz_corn[..., 1].max()
        z_min = xyz_corn[..., 2].min()
        z_max = xyz_corn[..., 2].max()
        dx = np.full(dimens, (x_max - x_min) / nx)
        dy = np.full(dimens, (y_max - y_min) / ny)
        dz = np.full(dimens, (z_max - z_min) / nz)
        tops = np.full(dimens, z_min)
        tops[..., 1:] += np.cumsum(dz, axis=2)[..., :-1]
        mapaxes = np.array([x_min, y_min + 1, x_min, y_min, x_min + 1, y_min])
        grid = OrthogonalGrid(dimens=dimens, dx=dx, dy=dy, dz=dz,
                              tops=tops, mapaxes=mapaxes)
        return grid

    def crop_minimal_cube(self):
        """Crop to minimal cube containing active cells.

        Returns
        -------
        (grid, min_slices) : tuple
            New grid and slices for active cube.
        """
        min_slices = self.minimal_active_slices()
        actnum = self.actnum[min_slices]
        zcorn = self.zcorn[min_slices]
        min_coord_slices = tuple(slice(s.start, s.stop + 1) for s in min_slices[:2])
        coord = self.coord[min_coord_slices]

        grid = self.__class__(actnum=actnum, coord=coord,
                              dimens=actnum.shape,
                              zcorn=zcorn, mapaxes=self.mapaxes)
        return grid, min_slices

    def crop_minimal_grid(self, nz, fillna=None):
        """Create a new grid in a region between upper and bottom surfaces enclosing active cells.
        Actnum transfer should be made separately.

        Parameters
        ----------
        nz : int
            Third dimension of the cunstructed grid.
        fillna : scalar
            Filling value for nan coordinates.

        Returns
        -------
        (grid, grid_mask, z_top, z_bottom) : tuple
        """
        z_top, z_bottom = self.minimal_active_bounds()
        with warnings.catch_warnings():  # ignore comparison with np.nan
            warnings.simplefilter("ignore")
            grid_mask = ((self.zcorn[:, :, :, 0] >= np.expand_dims(z_top[:-1, :-1], -1)) &
                         (self.zcorn[:, :, :, 4] <= np.expand_dims(z_bottom[:-1, :-1], -1)))

        def _fillna(x):
            """Replace np.nan with value if value is given."""
            if not np.isnan(x):
                return x
            return x if fillna is None else fillna

        grid_points = np.stack([np.linspace(_fillna(start), _fillna(stop), nz + 1)
                                for start, stop in zip(z_top.ravel(), z_bottom.ravel())])
        grid_points = grid_points.reshape(z_top.shape + (nz + 1,))
        zcorn = np.stack([grid_points[:-1, :-1, :-1],  # 0
                          grid_points[1:, :-1, :-1],  # 1
                          grid_points[:-1, 1:, :-1],  # 2
                          grid_points[1:, 1:, :-1],  # 3
                          grid_points[:-1, :-1, 1:],  # 4
                          grid_points[1:, :-1, 1:],  # 5
                          grid_points[:-1, 1:, 1:],  # 6
                          grid_points[1:, 1:, 1:]], axis=-1)
        grid = self.__class__(coord=self.coord, dimens=zcorn.shape[:3],
                              zcorn=zcorn, mapaxes=self.mapaxes)
        return grid, grid_mask, z_top, z_bottom

    def to_corner_point(self):
        """Returns itself."""
        return self

    @property
    def as_corner_point(self):
        """Returns itself."""
        return self

    def create_vtk_locator(self, use_only_active=True, scaling=False, **kwargs):
        """Creates locator and mapping dictionary.

        Returns
        -------
        grid : `pyvista.core.pointset.UnstructuredGrid` object
        """
        _ = kwargs
        grid_params = {'use_only_active': use_only_active, 'cell_size': None, 'scaling': scaling}
        if self._vtk_grid is None or self._vtk_grid_params != grid_params:
            self.create_vtk_grid(**grid_params)

        geo_filter = vtk.vtkGeometryFilter()
        geo_filter.SetInputData(self._vtk_grid)
        geo_filter.Update()
        polydata = geo_filter.GetOutput()

        locator = vtk.vtkModifiedBSPTree()
        locator.SetDataSet(polydata)
        locator.AutomaticOn()
        locator.BuildLocator()

        self._vtk_locator = locator

    def point_inside_cell(self, point, cell_idx, tolerance=1e-8):
        """Determines whether point is inside cell or not.

        Returns
        -------
        inside : bool

        Notes
        -----
        Result might be inconsistent due to the stochastic nature of the algorithm.
        Decreasing the tolerance might help.
        """
        x = self.xyz[cell_idx[0], cell_idx[1], cell_idx[2]].copy()

        x[[2, 3]] = x[[3, 2]]
        x[[6, 7]] = x[[7, 6]]

        pts = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
               (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

        cube = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        polys = vtk.vtkCellArray()

        for i in range(8):
            points.InsertPoint(i, x[i])
        for i in range(6):
            polys.InsertNextCell(mk_vtk_id_list(pts[i]))

        cube.SetPoints(points)
        cube.SetPolys(polys)

        points = np.array([point])
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetDataTypeToDouble()
        vtk_pts.SetData(numpy_support.numpy_to_vtk(points, deep=1))
        # Make the poly data
        poly_pts_vtp = vtk.vtkPolyData()
        poly_pts_vtp.SetPoints(vtk_pts)

        enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
        enclosed_points_filter.SetTolerance(tolerance)
        enclosed_points_filter.SetSurfaceData(cube)
        enclosed_points_filter.SetInputData(poly_pts_vtp)
        enclosed_points_filter.Update()

        return enclosed_points_filter.IsInside(0)

    def faces_centers(self, cells_indices):
        """Calculate coordinates of cell faces centers.

        Parameters
        ----------
        cells_indices : List[np.ndarray]
            Indices of the cells.

        Returns
        -------
        np.ndarray
            coordinates of cell faces centers.
        """
        xyz = self.xyz[cells_indices[0], cells_indices[1], cells_indices[2]]
        centers = []
        for vertices in (
                (0, 2, 4, 6), # left
                (1, 3, 5, 7), # right
                (0, 1, 4, 5), # back
                (2, 3, 6, 7), # front
                (0, 1, 2, 3), # top
                (4, 5, 6, 7)  # bottom
            ):
            centers.append(xyz[..., vertices, :].mean(axis=-2))
        centers = np.array(centers)
        if centers.ndim == 3:
            return np.moveaxis(centers, (0, 1), (1, 0))
        return centers

    def cell_sizes(self, cell_indices):
        """Calculate approximate sizes of cells.

        Parameters
        ----------
        cells_indices : List[np.ndarray]
            Indices of the cells.

        Returns
        -------
        np.ndarray
            Sizes of cells.
        """
        faces_centers = self.faces_centers(cell_indices)
        n_cells = cell_indices[0].size if isinstance(cell_indices[0], np.ndarray) else 1
        sizes = np.zeros((n_cells, 3))
        for i in range(3):
            sizes[:, i] = np.sqrt((((faces_centers[..., 2*i+1, :] - faces_centers[..., 2*i+0, :])**2)).sum(axis=-1))
        return sizes

    def cell_bases(self, cell_indices):
        """Calculate basis vectors of coordinate systems connected to cells.

        Parameters
        ----------
        cells_indices : List[np.ndarray]
            Indices of the cells.

        Returns
        -------
        np.ndarray
            Basis vectors of coordinate systems connected to cells.
        """
        faces_centers = self.faces_centers(cell_indices)
        i = faces_centers[..., 1, :] - faces_centers[..., 0, :]
        j = faces_centers[..., 3, :] - faces_centers[..., 2, :]
        z = np.cross(i, j)
        hat3 = (z.T / np.linalg.norm(z, axis=-1)).T
        x = i + np.cross(j, hat3)
        y = j - np.cross(i, hat3)
        hat1 = (x.T / np.linalg.norm(x, axis=-1)).T
        hat2 = (y.T / np.linalg.norm(y, axis=-1)).T
        return np.moveaxis(np.stack((hat1, hat2, hat3)), (0, 1), (1, 0))

    def map_grid(self):
        """Map pillars (`COORD`) to axis defined by `MAPAXES'.

        Returns
        -------
        CornerPointGrid
            Grid with updated `COORD` and `MAPAXES` fields.

        """
        if not np.isclose(self.ex.dot(self.ey), 0):
            raise ValueError('`ex` and `ey` vectors should be orthogonal.')

        new_basis = np.vstack((self.ex, self.ey)).T
        self.coord[:, :, :2] = self.coord[:, :, :2].dot(new_basis) + self.origin[:2]
        self.coord[:, :, 3:5] = self.coord[:, :, 3:5].dot(new_basis) + self.origin[:2]
        setattr(self, 'MAPAXES', np.array([0, 1, 0, 0, 1, 0]))
        return self

def specify_grid(grid):
    """Specify grid class: `CornerPointGrid` or `OrthogonalGrid`.

    Parameters
    ----------
    grid : Grid
        Initial grid.

    Returns
    -------
    CornerPointGrid or OrthogonalGrid
        specified grid.
    """
    if not isinstance(grid, (CornerPointGrid, OrthogonalGrid)):
        if ('DX' in grid) and ('DY' in grid) and ('DZ' in grid):
            grid = OrthogonalGrid(**dict(grid.items()), field=grid.field)
        else:
            grid = CornerPointGrid(**dict(grid.items()), field=grid.field)
    return grid
