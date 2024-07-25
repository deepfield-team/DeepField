"""Methods to update information about grid blocks for a well segment."""
import warnings
import numpy as np
from ..grids import OrthogonalUniformGrid

def calculate_cf(rock, grid, segment, beta=1, units='METRIC', cf_aggregation='sum'):
    """Calculate connection factor values for each grid block of a segment."""
    x_blocks, y_blocks, z_blocks = segment.blocks.T
    blocks_size = len(x_blocks)
    try:
        perm_tensor = np.vstack((rock.permx[x_blocks, y_blocks, z_blocks],
                                 rock.permy[x_blocks, y_blocks, z_blocks],
                                 rock.permz[x_blocks, y_blocks, z_blocks]))
    except AttributeError:
        return segment
    if units == 'METRIC':
        conversion_const = 0.00852702
    else:
        conversion_const = 0.00112712
    if isinstance(grid, OrthogonalUniformGrid):
        d_block = np.array([grid.cell_size]*blocks_size).T
    else:
        d_block = grid.cell_sizes((x_blocks, y_blocks, z_blocks)).T

    if 'CF' in segment.blocks_info.columns:
        ind = np.isnan(segment.blocks_info.CF.values)
    else:
        ind = np.arange(segment.blocks.shape[0])
    h_well = (segment.blocks_info[['Hx', 'Hy', 'Hz']].values.T *
              segment.blocks_info['PERF_RATIO'].values)
    r_well = segment.blocks_info['RAD'].values
    skin = segment.blocks_info['SKIN'].values
    wpi_mult = segment.blocks_info['MULT'].values
    if beta == 1:
        beta = np.array([beta]*blocks_size)
    else:
        beta = np.array(beta)
    d_1, d_2 = d_block[[1, 2, 0]], d_block[[2, 0, 1]]
    k_1, k_2 = perm_tensor[[1, 2, 0]], perm_tensor[[2, 0, 1]]
    k_h = (k_1 * k_2 * h_well**2)**0.5
    with warnings.catch_warnings(): # ignore devision by zero
        warnings.simplefilter("ignore")
        radius_equiv = (0.28 * (d_1**2 * np.sqrt(k_2 / k_1) + d_2**2 * np.sqrt(k_1 / k_2))**0.5 /
                        ((k_2 / k_1)**0.25 + (k_1 / k_2)**0.25))
        cf_projections = ((beta * wpi_mult * 2 * np.pi * conversion_const * k_h) /
                          (np.log(radius_equiv / r_well) + skin)).T
        if cf_aggregation == 'sum':
            segment.blocks_info.loc[ind, 'CF'] = cf_projections.sum(axis=1)
        elif cf_aggregation == 'eucl':
            segment.blocks_info.loc[ind, 'CF'] = np.sqrt((cf_projections ** 2).sum(axis=1))
        else:
            raise ValueError('Wrong value cf_aggregation={}, should be "sum" or "eucl".'.format(cf_aggregation))
    segment.blocks_info['CF'] = segment.blocks_info['CF'].fillna(0)
    return segment

def apply_perforations(segment, current_date=None):
    """Calculate perforation ratio for each grid block of the segment.

    ATTENTION: only latest perforation that covers the block
    defines the perforation ratio of this block.
    """
    if 'COMPDAT' in segment.attributes or ('COMPDATL' in segment.attributes and
                                           (segment.compdatl['LGR']=='GLOBAL').all()):
        return apply_perforations_compdat(segment, current_date)
    if 'COMPDATMD' in segment.attributes:
        mode = 'COMPDATMD'
    else:
        mode = 'PERF'
    if mode == 'COMPDATMD':
        perf = segment.compdatmd
    elif mode == 'PERF':
        perf = segment.perf
    else:
        raise ValueError(f'`mode` shoud be "PERF" or "COMPDATMD" not "{mode}"')
    if current_date is not None:
        perf = perf.loc[segment.perf['DATE'] < current_date]

    col_rad = 'DIAM' if 'DIAM' in perf.columns else 'RAD'

    b_info = segment.blocks_info
    b_info['PERF_RATIO'] = 0

    blocks_start_md = b_info['MD'].values
    last_block_size = np.linalg.norm(b_info.tail(1)[['Hx', 'Hy', 'Hz']])
    blocks_end_md = np.hstack([blocks_start_md[1:],
                               [blocks_start_md[-1] + last_block_size]])
    blocks_size = blocks_end_md - blocks_start_md
    columns = (['MDL', 'MDU', col_rad, 'SKIN', 'MULT', 'CLOSE'] if mode =='PERF' else
               ['MDU', 'MDL', col_rad, 'SKIN', 'MULT', 'MODE'])

    for line in perf[columns].values:
        md_start, md_end, rad, skin, wpimult, close_flag = line
        if mode == 'COMPDATMD':
            close_flag = close_flag!='OPEN'
        is_covered = (blocks_end_md > md_start) & (blocks_start_md <= md_end)
        if not is_covered.any():
            continue
        b_info.loc[is_covered, 'SKIN'] = skin
        b_info.loc[is_covered, 'MULT'] = wpimult
        b_info.loc[is_covered, 'RAD'] = rad/2 if 'DIAM' in perf.columns else rad
        covered_ids = np.where(is_covered)[0]
        if close_flag:
            b_info.loc[covered_ids, 'PERF_RATIO'] = 0
            continue
        first = covered_ids.min()
        last = covered_ids.max()
        #full perforation of intermediate blocks
        b_info.loc[first+1:last, 'PERF_RATIO'] = 1
        if first == last:
            #partial perforation of the single block
            perf_size = md_end - md_start
            b_info.loc[first, 'PERF_RATIO'] = min(perf_size / blocks_size[first], 1)
            continue
        #partial perforation of the first block
        perf_size = blocks_end_md[first] - md_start
        b_info.loc[first, 'PERF_RATIO'] = min(perf_size / blocks_size[first], 1)
        #partial perforation of the last block
        perf_size = md_end - blocks_start_md[last]
        b_info.loc[last, 'PERF_RATIO'] = min(perf_size / blocks_size[last], 1)
    return segment

def apply_perforations_compdat(segment, current_date=None):
    """Update `blocks_info`apply_perforations_compdat table with perforations parameters from `COMPDAT`
    or `COMPDATL`."""
    if current_date is None:
        if 'COMPDAT' in segment.attributes:
            compdat = segment.compdat
        elif 'COMPDATL' in segment.attributes:
            if (segment.compdatl['LGR'] != 'GLOBAL').any():
                raise ValueError('LGRs other than `GLOBAL` are not supported.')
            compdat = segment.compdatl
        else:
            raise ValueError('Segment has no COMPDAT or COMPDATL tables.')
    else:
        if 'COMPDAT' in segment.attributes:
            compdat = segment.compdat.loc[segment.compdat['DATE'] < current_date]
        elif 'COMPDATL' in segment.attributes:
            if (segment.compdatl['LGR'] != 'GLOBAL').any():
                raise ValueError('LGRs other than `GLOBAL` are not supported.')
            compdat = segment.loc[segment.compdatl['DATE'] < current_date]
        else:
            raise ValueError('Segment has no COMPDAT or COMPDATL tables.')

    cf = np.full(segment.blocks.shape[0], np.nan)
    skin = np.full(segment.blocks.shape[0], np.nan)
    perf_ratio = np.zeros(segment.blocks.shape[0])
    mult = np.full(segment.blocks.shape[0], np.nan)
    rad = np.full(segment.blocks.shape[0], np.nan)

    for i, line in compdat.iterrows():
        condition = np.where(
            np.logical_and(
                np.logical_and(segment.blocks[:, 0] == line['I'] - 1,
                               segment.blocks[:, 1] == line['J'] - 1),
                np.logical_and(segment.blocks[:, 2] >= line['K1'] - 1,
                               segment.blocks[:, 2] <= line['K2'] - 1)))
        if line['MODE'] == 'SHUT':
            close_flag = True
        elif line['MODE'] == 'OPEN':
            close_flag = False
        else:
            raise ValueError(('Incorrect mode `{}` in line {} of COMPDAT ' +
                              'for well `{}`').format(
                                  line['MODE'], i, segment.name))
        perf_ratio[condition] = 0 if close_flag else 1
        skin[condition] = line['SKIN'] if 'SKIN' in line else 0
        cf[condition] = line['CF'] if 'CF' in line else np.nan
        mult[condition] = line['MULT'] if 'MULT' in line else 1
        rad[condition] = line['DIAM'] / 2

    segment.blocks_info['PERF_RATIO'] = perf_ratio
    if 'CF' in compdat.columns:
        segment.blocks_info['CF'] = cf
    segment.blocks_info['SKIN'] = skin
    segment.blocks_info['MULT'] = mult
    segment.blocks_info['RAD'] = rad
    return segment
