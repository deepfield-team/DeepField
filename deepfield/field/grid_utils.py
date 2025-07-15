"""Grid utils."""
import numpy as np
from numba import njit

_SHIFTS = {
    (1, 0): (1, 0, 0),
    (2, 0): (0, 1, 0),
    (2, 1): (-1, 1, 0),
    (3, 0): (1, 1, 0),
    (3, 1): (0, 1, 0),
    (3, 2): (1, 0, 0),
    (4, 0): (0, 0, 1),
    (4, 1): (-1, 0, 1),
    (4, 2): (0, -1, 1),
    (4, 3): (-1, -1, 1),
    (5, 0): (1, 0, 1),
    (5, 1): (0, 0, 1),
    (5, 2): (1, -1, 1),
    (5, 3): (0, -1, 1),
    (5, 4): (1, 0, 0),
    (6, 0): (0, 1, 1),
    (6, 1): (-1, 1, 1),
    (6, 2): (0, 0, 1),
    (6, 3): (-1, 0, 1),
    (6, 4): (0, 1, 0),
    (6, 5): (-1, 1, 0),
    (7, 0): (1, 1, 1),
    (7, 1): (0, 1, 1),
    (7, 2): (1, 0, 1),
    (7, 3): (0, 0, 1),
    (7, 4): (1, 1, 0),
    (7, 5): (0, 1, 0),
    (7, 6): (1, 0, 0)
}

def default_connectivity_tensor(nx, ny, nz):
    """Default connectivity tensor."""
    indices = np.indices((nx, ny, nz))
    connectivity = np.zeros((nx, ny, nz, 8), int)
    connectivity[:, :, :, 0] = indices[2] * (ny+1)*(nx+1) + indices[1]*(nx+1) + indices[0]
    connectivity[:, :, :, 1] = connectivity[..., 0] + 1
    connectivity[:, :, :, 2] = connectivity[..., 0] + (nx+1)
    connectivity[:, :, :, 3] = connectivity[..., 2] + 1
    connectivity[:, :, :, 4:8] = connectivity[..., 0:4] + (ny+1)*(nx+1)
    return connectivity

def common_indices(indices, shape):
    """Mask common indices."""
    mask = np.zeros(shape, dtype=bool)
    mask[*indices[0]] = True
    for ind in indices[1:]:
        mask_tmp = np.zeros(shape, dtype=bool)
        mask_tmp[*ind] = True
        mask = np.logical_and(mask, mask_tmp)
    return np.where(mask)

def process_grid(zcorn, coord, actnum, active_points=True):
    """Get points and connectivity arrays for vtk grid."""

    def _get_slices(sh, size):
        x0 = 1 if sh==1 else 0
        x1 = size-1 if sh==-1 else size 
        return slice(x0, x1)

    def _process_indices(indices, shifts):
        return [(ind + 1 if sh==-1 else ind) for ind, sh in zip(indices, shifts)]

    def _get_default_mask(shifts, size, i):
        mask_tmp = np.zeros(size, dtype=bool)
        if shifts[0] == 1:
            mask_tmp[-1, :, :] = True
        elif shifts[0] == -1:
            mask_tmp[0, :, :] = True
        if shifts[1] == 1:
            mask_tmp[:, -1, :] = True
        elif shifts[1] == -1:
            mask_tmp[:, 0, :] = True
        if shifts[2] == 1:
            mask_tmp[:, :, -1] = True
        elif shifts[2] == -1:
            mask_tmp[:, :, 0] = True
        if i == 1:
            mask_tmp[-1, :, :] = False
        elif i == 2:
            mask_tmp[:, -1, :] = False
        elif i == 3:
            mask_tmp[-1, -1, :] = False
        elif i == 4:
            mask_tmp[:, :, -1] = False
        elif i == 5:
            mask_tmp[-1, :, -1] = False
        elif  i == 6:
            mask_tmp[:, -1, -1] = False
        elif i == 7:
            mask_tmp[-1, -1, -1] = False
        return mask_tmp

    nx, ny, nz = zcorn.shape[:3]
    points = np.zeros((nx+1, ny+1, nz+1), dtype=int)
    indices = np.indices((nx, ny, nz))

    points[:-1, :-1, :-1] = np.ravel_multi_index(
        [*indices, np.full((nx, ny, nz), 0)], (nx, ny, nz, 8))
    points[-1, :-1, :-1] = np.ravel_multi_index(
        [*indices[:, -1, :, :], np.full((ny, nz), 1)], (nx, ny, nz, 8))
    points[:-1, -1, :-1] = np.ravel_multi_index(
        [*indices[:, :, -1, :], np.full((nx, nz), 2)], (nx, ny, nz, 8))
    points[-1, -1, :-1] = np.ravel_multi_index(
        [*indices[:, -1, -1, :], np.full((nz,), 3)], (nx, ny, nz, 8))
    points[:-1, :-1, -1] = np.ravel_multi_index(
        [*indices[:, :, : , -1], np.full((nx, ny), 4)], (nx, ny, nz, 8))
    points[-1, :-1, -1] = np.ravel_multi_index(
        [*indices[:, -1, :, -1], np.full(ny, 5)], (nx, ny, nz, 8))
    points[:-1, -1, -1] = np.ravel_multi_index(
        [*indices[:, :, -1, -1], np.full(nx, 6)], (nx, ny, nz, 8))
    points[-1, -1, -1] = np.ravel_multi_index(
        [*indices[:, -1, -1, -1], 7], (nx, ny, nz, 8))

    points = points.reshape(-1, order='F')

    connectivity = default_connectivity_tensor(nx, ny, nz)

    n_nodes = (nx+1) * (ny+1) * (nz+1)

    for i in range(1, 8):
        ind_dif = []
        for j in range(i):
            shifts = _SHIFTS[(i, j)]
            slices0 = [_get_slices(sh, s) for sh, s in zip(shifts, (nx, ny, nz))]
            slices1 = [_get_slices(-sh, s) for sh, s in zip(shifts, (nx, ny, nz))]
            ind_tmp = _process_indices(
                np.where(zcorn[*slices1, i] != zcorn[*slices0, j]), shifts)
            mask_tmp = _get_default_mask(shifts, (nx, ny, nz), i)
            mask_tmp[*ind_tmp] = True
            ind_dif.append(np.where(mask_tmp))
        nodes_to_add = common_indices((ind_dif), (nx, ny, nz))
        points_to_add = np.ravel_multi_index(
            [*nodes_to_add, np.full(nodes_to_add[0].shape,i)], (nx, ny, nz, 8))
        n_nodes_to_add = nodes_to_add[0].size
        connectivity[*nodes_to_add, i] = n_nodes + np.arange(n_nodes_to_add)
        n_nodes = n_nodes + n_nodes_to_add
        points = np.concatenate((points, points_to_add), axis=0)
 
        for j in range(i+1, 8):
            shifts = _SHIFTS[(j, i)]
            slices0 = [_get_slices(sh, s) for sh, s in zip(shifts, (nx, ny, nz))]
            slices1 = [_get_slices(-sh, s) for sh, s in zip(shifts, (nx, ny, nz))]
            ind_to_repl0= _process_indices(
                np.where(zcorn[*slices1, j] == zcorn[*slices0, i]), shifts)
            ind_to_repl1 = _process_indices(
                np.where(zcorn[*slices1, j] == zcorn[*slices0, i]), [-s for s in shifts])
            connectivity[*ind_to_repl0, j] = connectivity[*ind_to_repl1, i]

    connectivity[:, :, :, [2, 3]] = connectivity[:, :, :, [3, 2]]
    connectivity[:, :, :, [6, 7]] = connectivity[:, :, :, [7, 6]]

    indices = np.unravel_index(points, (nx, ny, nz, 8))
    coord_indices = np.empty((2, points.size), dtype=int)

    coord_indices[0] = indices[0] + indices[3] % 2
    coord_indices[1] = indices[1] + ((indices[3] % 4) >= 2).astype(int)
    points = np.empty((coord_indices.shape[1], 3))
    points[:, 2] = zcorn[indices]
    coord_points = coord[*coord_indices]
    points[:, [0, 1]] = (coord_points[:, [0, 1]] +
                         (coord_points[:, [3, 4]] - coord_points[:, [0, 1]]) * 
                         ((points[:, 2] - coord_points[:, 2]) /
                         (coord_points[:, 5] - coord_points[:, 2]))[..., np.newaxis])

    return points, connectivity[actnum]

def process_grid_orthogonal(tops, dx, dy, dz, actnum):
    """Get points and connectivity arrays for orthogonal vtk grid."""
    nx, ny, nz = tops.shape

    connectivity = default_connectivity_tensor(nx, ny, nz)
    if (tops != tops[0:1, 0:1, :]).any():
        raise ValueError('`tops` values shoud be consistent for each layer.')
    if not (dx == dx[0, 0, 0]).all():
        raise ValueError('All `dx` values should be the same.')
    if not (dy == dy[0, 0, 0]).all():
        raise ValueError('All `dy` values should be the same.')
    if not (dz == dz[0, 0, 0]).all():
        raise ValueError('All `dz` values should be the same.')
    if not (tops[:, :, 1:] == tops[:, :, 0:1] + np.cumsum(dz, axis=2)[:, :, :-1]).all():
        raise ValueError('`tops` should be consistent with dz.')
    points = np.zeros((nx+1, ny+1, nz+1, 3), dtype=float)
    points[:, :, :, 0] = np.linspace(0, dx[0, 0, 0]*nx, (nx+1))[:, np.newaxis, np.newaxis]
    points[:, :, :, 1] = np.linspace(0, dy[0, 0, 0]*ny, (ny+1))[np.newaxis, :, np.newaxis]
    points[:, :, :, 2] = np.linspace(0, dz[0, 0, 0]*nz, (nz+1))[np.newaxis, np.newaxis, :]
    points = points.reshape(-1, 3, order='F')

    return points, connectivity[actnum]    

@njit
def isclose(a, b, rtol=1e-05, atol=1e-08):
    """np.isclose."""
    return abs(a - b) <= (atol + rtol * abs(b))

@njit
def reshape(i, j, k, nx, ny):
    """Ravel index."""
    return k*nx*ny + j*nx + i

@njit
def calc_point(i, j, z, n, coord):
    """Compute xyz from COORD."""
    if n in [0, 4]:
        ik, jk = i, j
    elif n in [1, 5]:
        ik, jk = i+1, j
    elif n in [2, 6]:
        ik, jk = i, j+1
    else:
        ik, jk = i+1, j+1

    line = coord[ik, jk]
    top_point = list(line[:3])
    vec = list(line[3:] - line[:3])
    is_degenerated = False
    if isclose(vec[2], 0):
        if not isclose(vec[0], 0) & isclose(vec[1], 0):
            vec[2] = 1e-10
        else:
            is_degenerated = True
    if is_degenerated:
        return [top_point[0], top_point[1], z]

    z_along_line = (z - top_point[2]) / vec[2]
    return [top_point[0] + vec[0]*z_along_line, top_point[1] + vec[1]*z_along_line, z]

@njit
def calc_cells(zcorn, coord):
    """Get points and connectivity arrays for vtk grid."""
    points = []
    conn = []

    a = zcorn
    nx, ny, nz, _ = a.shape

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn.append([0]*8)
                #0
                if a[i, j, k, 0] == a[i-1, j, k, 1] and i>0:
                    conn[-1][0] = conn[reshape(i-1, j, k, nx, ny)][1]
                elif a[i, j, k, 0] == a[i, j-1, k, 2] and j>0:
                    conn[-1][0] = conn[reshape(i, j-1, k, nx, ny)][2]
                elif a[i, j, k, 0] == a[i-1, j-1, k, 3] and i>0 and j>0:
                    conn[-1][0] = conn[reshape(i-1, j-1, k, nx, ny)][3]
                elif a[i, j, k, 0] == a[i, j, k-1, 4] and k>0:
                    conn[-1][0] = conn[reshape(i, j, k-1, nx, ny)][4]
                elif a[i, j, k, 0] == a[i-1, j, k-1, 5] and i>0 and k>0:
                    conn[-1][0] = conn[reshape(i-1, j, k-1, nx, ny)][5]
                elif a[i, j, k, 0] == a[i, j-1, k-1, 6] and j>0 and k>0:
                    conn[-1][0] = conn[reshape(i, j-1, k-1, nx, ny)][6]
                elif a[i, j, k, 0] == a[i-1, j-1, k-1, 7] and i>0 and j>0 and k>0:
                    conn[-1][0] = conn[reshape(i-1, j-1, k-1, nx, ny)][7]
                else:
                    z = a[i, j, k, 0]
                    points.append(calc_point(i, j, z, 0, coord))
                    conn[-1][0] = len(points)-1

                #1
                if i<nx-1 and j>0 and a[i, j, k, 1] == a[i+1, j-1, k, 2]:
                    conn[-1][1] = conn[reshape(i+1, j-1, k, nx, ny)][2]
                elif a[i, j, k, 1] == a[i, j-1, k, 3] and j>0:
                    conn[-1][1] = conn[reshape(i, j-1, k, nx, ny)][3]
                elif i<nx-1 and k>0 and a[i, j, k, 1] == a[i+1, j, k-1, 4]:
                    conn[-1][1] = conn[reshape(i+1, j, k-1, nx, ny)][4]
                elif a[i, j, k, 1] == a[i, j, k-1, 5] and k>0:
                    conn[-1][1] = conn[reshape(i, j, k-1, nx, ny)][5]
                elif a[i, j, k, 1] == a[i-1, j-1, k-1, 6] and i>0 and j>0 and k>0:
                    conn[-1][1] = conn[reshape(i-1, j-1, k-1, nx, ny)][6]
                elif a[i, j, k, 1] == a[i, j-1, k-1, 7] and j>0 and k>0:
                    conn[-1][1] = conn[reshape(i, j-1, k-1, nx, ny)][7]
                else:
                    z = a[i, j, k, 1]
                    points.append(calc_point(i, j, z, 1, coord))
                    conn[-1][1] = len(points)-1

                #2
                if a[i, j, k, 2] == a[i-1, j, k, 3] and i>0:
                    conn[-1][2] = conn[reshape(i-1, j, k, nx, ny)][3]
                elif j<ny-1 and k>0 and a[i, j, k, 2] == a[i, j+1, k-1, 4]:
                    conn[-1][2] = conn[reshape(i, j+1, k-1, nx, ny)][4]
                elif i>0 and j<ny-1 and k>0 and a[i, j, k, 2] == a[i-1, j+1, k-1, 5]:
                    conn[-1][2] = conn[reshape(i-1, j+1, k-1, nx, ny)][5]
                elif a[i, j, k, 2] == a[i, j, k-1, 6] and k>0:
                    conn[-1][2] = conn[reshape(i, j, k-1, nx, ny)][6]
                elif a[i, j, k, 2] == a[i-1, j, k-1, 7] and i>0 and k>0:
                    conn[-1][2] = conn[reshape(i-1, j, k-1, nx, ny)][7]
                else:
                    z = a[i, j, k, 2]
                    points.append(calc_point(i, j, z, 2, coord))
                    conn[-1][2] = len(points)-1

                #3
                if i<nx-1 and j<ny-1 and k>0 and a[i, j, k, 3] == a[i+1, j+1, k-1, 4]:
                    conn[-1][3] = conn[reshape(i+1, j+1, k-1, nx, ny)][4]
                elif j<ny-1 and k>0 and a[i, j, k, 3] == a[i, j+1, k-1, 5]:
                    conn[-1][3] = conn[reshape(i, j+1, k-1, nx, ny)][5]
                elif i<nx-1 and k>0 and a[i, j, k, 3] == a[i+1, j, k-1, 6]:
                    conn[-1][3] = conn[reshape(i+1, j, k-1, nx, ny)][6]
                elif a[i, j, k, 3] == a[i, j, k-1, 7] and k>0:
                    conn[-1][3] = conn[reshape(i, j, k-1, nx, ny)][7]
                else:
                    z = a[i, j, k, 3]
                    points.append(calc_point(i, j, z, 3, coord))
                    conn[-1][3] = len(points)-1

                #4
                if a[i, j, k, 4] == a[i-1, j, k, 5] and i>0:
                    conn[-1][4] = conn[reshape(i-1, j, k, nx, ny)][5]
                elif a[i, j, k, 4] == a[i, j-1, k, 6] and j>0:
                    conn[-1][4] = conn[reshape(i, j-1, k, nx, ny)][6]
                elif a[i, j, k, 4] == a[i-1, j-1, k, 7] and i>0 and j>0:
                    conn[-1][4] = conn[reshape(i-1, j-1, k, nx, ny)][7]
                else:
                    z = a[i, j, k, 4]
                    points.append(calc_point(i, j, z, 4, coord))
                    conn[-1][4] = len(points)-1

                #5
                if i<nx-1 and j>0 and a[i, j, k, 5] == a[i+1, j-1, k, 6]:
                    conn[-1][5] = conn[reshape(i+1, j-1, k, nx, ny)][6]
                elif a[i, j, k, 5] == a[i, j-1, k, 7] and j>0:
                    conn[-1][5] = conn[reshape(i, j-1, k, nx, ny)][7]
                else:
                    z = a[i, j, k, 5]
                    points.append(calc_point(i, j, z, 5, coord))
                    conn[-1][5] = len(points)-1

                #6
                if a[i, j, k, 6] == a[i-1, j, k, 7] and i>0:
                    conn[-1][6] = conn[reshape(i-1, j, k, nx, ny)][7]
                else:
                    z = a[i, j, k, 6]
                    points.append(calc_point(i, j, z, 6, coord))
                    conn[-1][6] = len(points)-1

                #7
                z = a[i, j, k, 7]
                points.append(calc_point(i, j, z, 7, coord))
                conn[-1][7] = len(points)-1

    return points, conn

@njit
def get_xyz(dimens, zcorn, coord):
    """Get x, y, z coordinates of cell vertices."""
    nx, ny, _ = dimens
    xyz = np.zeros(zcorn.shape[:3] + (8, 3))
    xyz[..., 2] = zcorn
    shifts = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
    for i in range(nx + 1):
        for j in range(ny + 1):
            line = coord[i, j]
            top_point = line[:3]
            vec = line[3:] - line[:3]
            is_degenerated = False
            if isclose(vec[2], 0):
                if not isclose(vec[0], 0) & isclose(vec[1], 0):
                    vec[2] = 1e-10
                else:
                    is_degenerated = True

            for k in range(8):
                ik = i + shifts[k % 4][0]
                jk = j + shifts[k % 4][1]
                if (ik < 0) or (ik >= nx) or (jk < 0) or (jk >= ny):
                    continue
                if is_degenerated:
                    xyz[ik, jk, :, k] = top_point
                else:
                    z_along_line = (zcorn[ik, jk, :, k] - top_point[2]) / vec[2]
                    xyz[ik, jk, :, k, :2] = top_point[:2] + vec[:2] * z_along_line.reshape((-1, 1))
    return xyz

@njit
def get_xyz_ijk(zcorn, coord, ijk):
    """Get x, y, z coordinates of cell vertices for cells at ijk positions."""
    ijk = np.asarray(ijk).reshape(-1, 3)
    xyz = np.zeros((len(ijk),) + (8, 3))
    for p, (i,j,k) in enumerate(ijk):
        for n in range(8):
            z = zcorn[i, j, k, n]
            xyz[p, n] = calc_point(i, j, z, n, coord)
    return xyz

@njit
def get_xyz_ijk_orth(dx, dy, dz, tops, origin, ijk):
    """Get x, y, z coordinates of cell vertices for cells at ijk positions in orthogonal grid."""
    ijk = np.asarray(ijk).reshape(-1, 3)
    xyz = np.zeros((len(ijk),) + (8, 3))
    xyz[..., 0] = origin[0]
    xyz[..., 1] = origin[1]
    for p, (i,j,k) in enumerate(ijk):
        px = np.cumsum(dx[:, j, k])
        py = np.cumsum(dy[i, :, k])
        for t in [0, 2, 4, 6]:
            xyz[p, t, 0] = px[i]
        if i > 0:
            for t in [1, 3, 5, 7]:
                xyz[p, t, 0] = px[i-1]
        for t in [0, 1, 4, 5]:
            xyz[p, t, 1] = py[j]
        if j > 0:
            for t in [2, 3, 6, 7]:
                xyz[p, t, 1] = py[j-1]
        xyz[p, :4, 2] = tops[i, j, k]
        xyz[p, 4:, 2] = tops[i, j, k] + dz[i, j, k]
    return xyz
