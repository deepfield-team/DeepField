"""Classes and routines for handling model grids."""
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from .decorators import cached_property, apply_to_each_input
from .base_spatial import SpatialComponent
from .grid_utils import (get_xyz, get_xyz_ijk, get_xyz_ijk_orth,
                         process_grid, process_grid_orthogonal)
from .utils import rolling_window, get_single_path
from .parse_utils import read_ecl_bin


class Grid(SpatialComponent):
    """Basic grid class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vtk_grid = vtk.vtkUnstructuredGrid()
        self._vtk_locator = None
        self._actnum_ids = None
        self.to_spatial()
        if 'MAPAXES' not in self:
            setattr(self, 'MAPAXES', np.array([0, 1, 0, 0, 1, 0]))
        if 'ACTNUM' not in self and 'DIMENS' in self:
            self.actnum = np.ones(self.dimens, dtype=bool)

    @property
    def vtk_grid(self):
        """VTK unstructured grid."""
        return self._vtk_grid

    @property
    def locator(self):
        """VTK locator."""
        if self._vtk_locator is None:
            self.create_vtk_locator()
        return self._vtk_locator

    def create_vtk_grid(self):
        """Creates VTK instructured grid."""
        self._create_vtk_grid()
        self._vtk_locator = None
        return self

    def create_vtk_locator(self):
        """Creates VTK localor."""
        self._vtk_locator = vtk.vtkModifiedBSPTree()
        self._vtk_locator.SetDataSet(self.vtk_grid)
        self._vtk_locator.AutomaticOn()
        self._vtk_locator.BuildLocator()
        return self

    def _create_vtk_grid(self):
        """Create vtk grid from points and connectivity arrays."""
        points, conn = self.get_points_and_coonectivity()
        cell_array = vtk.vtkCellArray()

        for x in conn:
            cell_array.InsertNextCell(8, x)

        vtk_points = vtk.vtkPoints()
        for i, point in enumerate(points):
            vtk_points.InsertPoint(i, point)

        self.vtk_grid.SetPoints(vtk_points)
        self.vtk_grid.SetCells(vtk.vtkHexahedron().GetCellType(), cell_array)

        self._actnum_ids = np.where(self.actnum.ravel())[0]
        return self

    def get_points_and_coonectivity(self):
        """Get points and connectivity arrays."""
        raise NotImplementedError()

    def id_to_ijk(self, idx):
        """Convert raveled positional index of active cell to ijk."""
        idx = self.actnum_ids[np.asarray(idx)]
        return np.stack(np.unravel_index(idx, self.dimens), axis=-1)

    def ijk_to_id(self, ijk):
        """Convert ijk index of active cell to raveled positional index."""
        ids = []
        ijk = np.asarray(ijk).reshape(-1, 3)
        raveled = np.ravel_multi_index(ijk.T, self.dimens)
        for i, n in enumerate(raveled):
            try:
                ids.append(np.where(self.actnum_ids == n)[0][0])
            except IndexError as exc:
                raise IndexError("Can not compute index: cell ({}, {}, {}) is inactive.".format(*ijk[i])) from exc
        return ids

    @property
    def actnum_ids(self):
        """Raveled indices of active cells."""
        return self._actnum_ids


    def get_xyz(self, ijk=None):
        """Get x, y, z coordinates of cell vertices."""
        raise NotImplementedError()

    @property
    def origin(self):
        """Grid axes origin relative to the map coordinates."""
        return np.array([self.mapaxes[2], self.mapaxes[3], self.tops.ravel()[0]])

    @property
    def cell_centroids(self):
        """Centroids of cells."""
        filt = vtk.vtkCellCenters()
        filt.SetInputDataObject(self.vtk_grid)
        filt.Update()
        return vtk_to_numpy(filt.GetOutput().GetPoints().GetData())

    @property
    def cell_volumes(self):
        """Volumes of cells."""
        filt = vtk.vtkCellSizeFilter()
        filt.ComputeVolumeOn()
        filt.SetInputDataObject(self.vtk_grid)
        filt.Update()
        return vtk_to_numpy(filt.GetOutput().GetCellData().GetArray("Volume"))

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
        bounds = self.vtk_grid.GetBounds()
        return np.hstack([bounds[::2], bounds[1::2]])

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
        poro = self.field.rock.poro.ravel()[self.actnum_ids]
        if 'NTG' in self.field.rock:
            ntg = self.field.rock.ntg.ravel()[self.actnum_ids]
        else:
            ntg = 1
        mask = poro*volumes*ntg >= minpv_value
        if not mask.all():
            new_actnum = np.full(self.actnum.size, False)
            new_actnum[self.actnum_ids[mask]] = True
            self.actnum = new_actnum.reshape(self.dimens)
            self.create_vtk_grid()

    @apply_to_each_input
    def _to_spatial(self, attr, **kwargs):
        """Spatial order 'F' transformations."""
        _ = kwargs
        data = getattr(self, attr)
        if isinstance(data, np.ndarray) and data.ndim == 1:
            if attr in ['ACTNUM', 'DX', 'DY', 'DZ']:
                data = data.reshape(self.dimens, order='F')
            elif attr == 'TOPS':
                if data.size == np.prod(self.dimens):
                    data = data.reshape(self.dimens, order='F')
                else:
                    data = data.reshape(self.dimens[:2], order='F')
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


class OrthogonalGrid(Grid):
    """Orthogonal uniform grid."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'TOPS' not in self and 'DZ' in self:
            tops = np.zeros(self.dimens)
            tops[..., 1:] = np.cumsum(self.dz, axis=-1)[..., :-1]
            setattr(self, 'TOPS', tops)
        elif self.tops.ndim == 2 and 'DZ' in self:
            tops = np.zeros(self.dimens)
            tops[..., 1:] = np.cumsum(self.dz, axis=-1)[..., :-1]
            tops += self.tops[:, :, None]
            setattr(self, 'TOPS', tops)

    def get_xyz(self, ijk=None):
        """Get x, y, z coordinates of cell vertices."""
        if ijk is None:
            xyz = np.zeros(tuple(self.dimens) + (8, 3))
            xyz[..., 0] = self.origin[0]
            xyz[..., 1] = self.origin[1]
            px = np.cumsum(self.dx, axis=0)
            py = np.cumsum(self.dy, axis=1)
            xyz[1:, :, :, [0, 2, 4, 6], 0] += px[:-1, :, :, None]
            xyz[:, :, :, [1, 3, 5, 7], 0] += px[..., None]
            xyz[:, 1:, :, [0, 1, 4, 5], 1] += py[:, :-1, :, None]
            xyz[:, :, :, [2, 3, 6, 7], 1] += py[..., None]
            xyz[:, :, :, :4, 2] = self.tops[..., None]
            xyz[:, :, :, 4:, 2] = (self.tops + self.dz)[..., None]
            return xyz
        return get_xyz_ijk_orth(self.dx, self.dy, self.dz,
                                self.tops, self.origin, ijk)

    def get_points_and_coonectivity(self):
        """Get points and connectivity arrays."""
        try:
            return process_grid_orthogonal(self.tops, self.dx, self.dy, self.dz, self.actnum)
        except ValueError:
            grid = self.to_corner_point()
            return grid.get_points_and_coonectivity()

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

        actnum = np.kron(self.actnum, np.ones(factors)).astype(bool)

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
        grid.create_vtk_grid()
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

    def get_xyz(self, ijk=None):
        "Get x, y, z coordinates of cell vertices."
        if ijk is None:
            return get_xyz(self.dimens, self.zcorn, self.coord)
        return get_xyz_ijk(self.zcorn, self.coord, ijk)

    def get_points_and_coonectivity(self):
        """Get points and connectivity arrays."""
        return process_grid(self.zcorn, self.coord, self.actnum)

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

    def to_corner_point(self):
        """Returns itself."""
        return self

    @property
    def as_corner_point(self):
        """Returns itself."""
        return self

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
