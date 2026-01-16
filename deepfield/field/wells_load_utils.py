"""Wells utils."""
from __future__ import annotations
import re
import shlex
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from  .wells import Wells

from .well_segment import WellSegment
from .utils import get_multout_paths, get_single_path
from .parse_utils import (read_rsm, parse_perf_line, parse_control_line,
                          parse_history_line, read_ecl_bin, parse_eclipse_keyword)

DEFAULTS = {'RAD': 0.1524, 'DIAM': 0.3048, 'SKIN': 0, 'MULT': 1, 'CLOSE': False,
            'MODE': 'OPEN', 'DIR': 'Z', 'GROUP': 'FIELD'}

MODE_CONTROL = ['PROD', 'INJE', 'STOP']

VALUE_CONTROL = ['BHPT', 'THPT', 'DRAW', 'ETAB', 'OPT', 'GPT', 'WPT', 'LPT', 'VPT',
                 'OIT', 'GIT', 'WIT',
                 'HOIL', 'HGAS', 'HWAT', 'HLIQ', 'HBHP', 'HTHP', 'HWEF',
                 'GOPT', 'GGPT', 'GWPT', 'GLPT',
                 'GGIT', 'GWIT',
                 'OIL', 'GAS', 'WAT', 'LIQ', 'BHP', 'THP', 'GOR', 'OGR', 'WCT', 'WOR', 'WGR',
                 'DREF'
                 ]

def load_rsm(wells, path, logger):
    """Load RSM well data from file."""
    logger.info("Start reading {}".format(path))
    rsm = read_rsm(path, logger)
    logger.info("Finish reading {}".format(path))
    if '_global' not in rsm:
        logger.warning("Empty RSM file {}".format(path))
        return wells
    if '_children' in rsm['_global']:
        del rsm['_global']['_children']
    df = pd.DataFrame({k: v['data'] for k, v in rsm['_global'].items()})
    dates = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    welldata = {}
    for wellname, data in rsm.items():
        if wellname == '_global':
            continue
        if '_children' in data:
            del data['_children']
        wellname = wellname.strip(' \t\'\"').upper()
        wdf = pd.DataFrame({k: v['data'] * v['multiplyer'] for k, v in data.items()})
        wdf['DATE'] = dates
        wdf = wdf[['DATE'] + [col for col in wdf.columns if col != 'DATE']]
        welldata[wellname] = {'RESULTS': wdf.sort_values('DATE')}
    return wells.update(welldata)

def load_ecl_binary(wells, path_to_results, basename, logger=None, **kwargs):
    """Load results from UNSMRY file."""
    _ = kwargs

    smry_path_unifout = get_single_path(path_to_results, basename + '.UNSMRY', logger)
    smry_path_multout = get_multout_paths(path_to_results, basename, r'S\d+')
    if smry_path_unifout is None and smry_path_multout is None:
        return wells

    spec_path = get_single_path(path_to_results, basename + '.SMSPEC', logger)
    if spec_path is None:
        return wells

    def is_well_name(s):
        return re.match(r"[a-zA-Z0-9]", s) is not None

    if smry_path_unifout:
        smry_data_tmp = read_ecl_bin(smry_path_unifout, attrs=['PARAMS'],
                                     sequential=True, logger=logger)['PARAMS']
    elif smry_path_multout:
        smry_data_tmp = [read_ecl_bin(
            path, attrs=['PARAMS'],
            sequential=True, logger=logger)['PARAMS'][0] for path in smry_path_multout]
    else:
        raise ValueError('Neither `summary_path_unifout` or `summary_path_multout` is defined.')
    smry_data = np.stack(smry_data_tmp) # type: ignore

    spec_dict = read_ecl_bin(spec_path, attrs=['KEYWORDS', 'WGNAMES'],
                             sequential=False, logger=logger)
    kw = [w.strip() for w in spec_dict['KEYWORDS']]
    wellnames = [w.strip() for w in spec_dict['WGNAMES']]

    df = pd.DataFrame({k: smry_data[:, kw.index(k)].astype(int)
                       for k in ['DAY', 'MONTH', "YEAR"]})
    dates = pd.to_datetime(df.YEAR*10000 + df.MONTH*100 + df.DAY, format='%Y%m%d')

    welldata = {w: {'RESULTS': pd.DataFrame({'DATE': dates})}
                for w in np.unique(wellnames) if is_well_name(w)}
    for i, w in enumerate(wellnames):
        if w not in welldata:
            continue
        welldata[w]['RESULTS'][kw[i]] = smry_data[:, i]
    for v in welldata.values():
        v['RESULTS'].sort_values('DATE', inplace=True)
    wells.state.binary_attributes.append('RESULTS')
    return wells.update(welldata)

def load_group(wells, buffer, **kwargs):
    """Load groups. Note: optional keyword FRAC is not implemented."""
    _ = kwargs
    group = next(iter(buffer)).upper().split('FRAC')[0].split()
    group_name = group[1]
    if group_name == '1*':
        group_name = DEFAULTS['GROUP']
    try:
        group_node = wells[group_name]
    except KeyError:
        group_node = WellSegment(parent=wells.root, name=group_name, ntype="group")
    for well in group[2:]:
        try:
            node = wells[well]
            node.parent = group_node
        except KeyError:
            WellSegment(parent=group_node, name=well)
    return wells

