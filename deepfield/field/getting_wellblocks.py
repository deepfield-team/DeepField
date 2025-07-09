"""Getting wellblocks."""
import numpy as np
from numba import njit
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

@njit
def point_in_box(point, bounding_box):
    """Check if point is inside the bounding box."""
    return (point >= bounding_box[:3]).all() and (point <= bounding_box[3:]).all()

@njit
def get_scale(vec, target_vec):
    """Scale a given vector to the target vector."""
    return min([abs(target_vec[i] / vec[i]) for i in range(3) if abs(vec[i])>0]) #pylint: disable=consider-using-generator

def get_wellblocks_vtk(welltrack, grid):
    """Get cells intersected by the welltrack and intersection info.

    Parameters
    ----------
    welltrack : array_like
        Well segment coordinates (x, y, z, md) from welltrack.
    grid : class instance
        Grid class.

    Returns
    -------
    blocks : ndarray
        Indices of grid cells intersected by the welltrack.
    points : ndarray
        Enter and leave points for each intersected cell.
    mds : ndarray
        MD for ender and leave point in each intersected cell.

    """
    locator = grid.locator

    points_intersection = vtk.vtkPoints()
    points_intersection.SetDataTypeToDouble()
    cells_intersection = vtk.vtkIdList()

    blocks = []
    points = []
    mds = []

    bounds = grid.bounding_box #xmin, ymin, zmin, xmax, ymax, zmax
    b_vec = bounds[3:] - bounds[:3]

    for i in range(len(welltrack)-1):
        md = welltrack[i][-1]
        a = welltrack[i][:3]
        b = welltrack[i+1][:3]

        seg_length = np.linalg.norm(b - a)
        if np.isclose(seg_length, 0):
            continue

        unit_vec = (b - a) / seg_length
        scale = get_scale(unit_vec, b_vec)

        a_inf = a - unit_vec*scale*1.01 if point_in_box(a, bounds) else a
        b_inf = b + unit_vec*scale*1.01 if point_in_box(b, bounds) else b

        code = locator.IntersectWithLine(a_inf, b_inf, 1e-10,
                                         points_intersection,
                                         cells_intersection)
        if not code:
            continue

        next_blocks = []
        for i in range(cells_intersection.GetNumberOfIds()):
            next_blocks.append(grid.id_to_ijk(cells_intersection.GetId(i)))

        enter_points = vtk_to_numpy(points_intersection.GetData()).copy()
        enter_dist = ((enter_points - a) * unit_vec).sum(axis=1)

        code = locator.IntersectWithLine(b_inf, a_inf, 1e-10,
                                         points_intersection,
                                         cells_intersection)
        if not code:
            continue

        next_blocks2 = []
        for i in range(cells_intersection.GetNumberOfIds()):
            next_blocks2.append(grid.id_to_ijk(cells_intersection.GetId(i)))
        next_blocks2 = np.array(next_blocks2)[::-1]

        order = []
        for i in range(len(next_blocks)): #pylint:disable=consider-using-enumerate)
            if (next_blocks[i] == next_blocks2[i]).all():
                order.append(i)
            else:
                j = np.where((next_blocks[i] == next_blocks2).all(axis=1))[0][0]
                order.append(j)

        next_blocks2 = next_blocks2[order]

        leave_points = vtk_to_numpy(points_intersection.GetData()).copy()[::-1]
        leave_points = leave_points[order]
        leave_dist = ((leave_points - a) * unit_vec).sum(axis=1)

        block_segments = np.stack([enter_points, leave_points], axis=1)

        block_segments[enter_dist < 0, 0] = a
        block_segments[leave_dist > seg_length, 1] = b

        block_length = np.linalg.norm(block_segments[:,1] - block_segments[:,0], axis=1)

        mask = np.isclose(block_length, 0) | (enter_dist >= seg_length) | (leave_dist < 0)
        if mask.all():
            continue

        next_blocks = np.array(next_blocks)[~mask]
        block_segments = block_segments[~mask]

        enter_md = md + np.linalg.norm(block_segments[:, 0] - a, axis=1)
        leave_md = md + np.linalg.norm(block_segments[:, 1] - a, axis=1)
        next_mds = np.stack([enter_md, leave_md], axis=1)

        if not blocks or (blocks[-1] != next_blocks[0]).any():
            blocks.extend(next_blocks)
            points.extend(block_segments)
            mds.extend(next_mds)
        else:
            blocks.extend(next_blocks[1:])
            points[-1][1] = block_segments[0][1]
            points.extend(block_segments[1:])
            mds[-1][1] = next_mds[0][1]
            mds.extend(next_mds[1:])

    blocks = np.array(blocks).reshape(-1, 3)
    points = np.array(points).reshape(-1, 2, 3) #pylint: disable=too-many-function-args
    mds = np.array(mds).reshape(-1, 2)

    return blocks, points, mds

def get_wellblocks_compdat(compdat):
    """Get wellblocks from `COMPDAT` or 'COMPDATL' table.

    Parameters
    ----------
    compdat : pandas.DataFrame
        `COMPDAT` table.

    Returns
    -------
    numpy.ndarray
        Block indices.
    """
    i = []
    j = []
    k = []
    for _, row in compdat.iterrows():
        k_row = list(range(int(row['K1']-1), int(row['K2'])))
        k += k_row
        i += [int(row['I'])-1] * len(k_row)
        j += [int(row['J'])-1] * len(k_row)
    return np.array(list(set((a, b, c) for a, b, c in zip(i, j, k))))
