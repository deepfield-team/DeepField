"""Grid utils."""
import numpy as np
from numba import njit


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
def numba_get_xyz(dimens, zcorn, coord):
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
