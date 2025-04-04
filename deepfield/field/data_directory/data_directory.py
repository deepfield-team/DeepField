from dataclasses import dataclass
from enum import Enum, auto
import os
from typing import Callable, Optional

import pandas as pd
import numpy as np

MAX_STRLEN = 40

INT_NAN = -99999999

FLUID_KEYWORDS = ('OIL', 'GAS', 'WATER')
ORTHOGONAL_GRID_KEYWORDS = ('DX', 'DY', 'DZ', 'TOPS', 'ACTNUM')
ROCK_GRID_KEYWORDS = ('PORO', 'PERMX', 'PERMY', 'PERMZ', 'NTG')
TABLES_KEYWORDS = ('DENSITY', 'PVCDO', 'PVTW', 'ROCK', 'SWOF')
FIELD_SUMMARY_KETWORDS = ('FOPR', 'FWPR', 'FWIR')
WELL_SUMMARY_KEYWORDS = ('WOPR', 'WWPR', 'WWIR', 'WLPR', 'WBHP')
TOTAL_SUMMARY_KEYWORDS = ('FOPT', 'FWPT', 'FWIT')
SCHEDULE_KEYWORDS = ('WELSPECS','COMPDAT', 'WCONPOD', 'WCONINJE')
DIMS_KEYWORDS = ('TABDIMS', 'EQLDIMS', 'REGDIMS', 'WELLDIMS', 'VFPPDIMS', 'VFPIDIMS',
                 'AQUDIMS')

_ATM_TO_PSI = 14.69

TABLE_INFO = {
    'PVTO': dict(attrs=['RS', 'PRESSURE', 'FVF', 'VISC'], domain=[0, 1],
                 defaults=None),

    'PVTG': dict(attrs=['PRESSURE', 'RV', 'FVF', 'VISC'], domain=[0, 1],
                 defaults=None),

    'PVDG': dict(attrs=['PRESSURE', 'FVF', 'VISC'], domain=[0],
                 defaults=None),

    'PVDO': dict(attrs=['PRESSURE', 'FVF', 'VISC'], domain=[0],
                 defaults=None),

    'PVTW': dict(attrs=['PRESSURE', 'FVF', 'COMPR', 'VISC', 'VISCOSIBILITY'], domain=[0],
                 defaults=[(1, _ATM_TO_PSI), 1, (4e-5, 4e-5/_ATM_TO_PSI), 0.3, 0]),

    'PVCDO': dict(attrs=['PRESSURE', 'FVF', 'COMPR', 'VISC', 'VISCOSIBILITY'], domain=[0],
                 defaults=[None, None, None, None, 0]),

    'SWOF': dict(attrs=['SW', 'KRWO', 'KROW', 'POW'], domain=[0],
                 defaults=[None, None, None, 0]),

    'SGOF': dict(attrs=['SG', 'KRGO', 'KROG', 'POG'], domain=[0],
                 defaults=[None, None, None, 0]),

    'RSVD': dict(attrs=['DEPTH', 'RS'], domain=[0],
                 defaults=None),

    'ROCK': dict(attrs=['PRESSURE', 'COMPR'], domain=[0],
                 defaults=[(1.0132, 1.0132*_ATM_TO_PSI), (4.934e-5, 4.934e-5/_ATM_TO_PSI)]),

    'DENSITY': dict(attrs=['DENSO', 'DENSW', 'DENSG'], domain=None,
                    defaults=[(600, 37.457),  (999.014, 62.366), (1, 0.062428)]),

    'EQUIL': dict(attrs=['DEPTH', 'PRES', 'WOC_DEPTH', 'WOC_PC', 'GOC_DEPTH', 'GOC_PC', 'RSVD_PBVD_TABLE',
                         'RVVD_PDVD_TABLE'], domain=None),

    'TABDIMS': dict(attrs=['SAT_REGIONS_NUM', 'PVT_REGIONS_NUM', 'SAT_NODES_NUM', 'PVT_NODES_NUM',
                           'FIP_REGIONS_NUM', 'OIL_VAP_NODES_NUM', 'OGR_NODES_NUM', 'SAT_END_POINT_NUM',
                           'EOS_REGIONS_NUM', 'EOS_SURFACE_REGIONS_NUM', 'FLUX_REGIONS_NUM', 'THERMAL_REGIONS_NUM',
                           'ROCK_TABLES_NUM', 'PRESSURE_MAINTAINACE_REGIONS_NUM', 'TEMPERATURE_NODES_NUM',
                           'TRANSPORT_COEFFICIENTS_NUM'
                           ], domain=None, dtype=int),

    'EQLDIMS': dict(attrs=['EQL_NUM', 'EQL_NODE_NUM', 'DEPTH_NODE_MAX_NUM', 'INIT_TRAC_CONC_NUM',
                           'INIT_TRAC_CONC_NODE_NUM'],
                    domain=None, dtype=int),

    'REGDIMS': dict(attrs=['FIP_REGIONS_NUM', 'FIP_FAMILIES_NUM', 'RESERVOIR_REGIONS_NUM',
                           'FLUX_REGIONS_NUM', 'TRACK_REGIONS_NUM', 'COAL_REG_NUM', 'OPER_REGIONS_NUM',
                           'WORK_NUM', 'IWORK_NUM', 'PLMIX_REGIONS_NUM'], domain=None, dtype=int),

    'WELLDIMS': dict(attrs=['WELL_NUM', 'CONN_NUM', 'GROUP_NUM', 'WELL_IN_GROUP_NUM', 'SEP_STAGES_NUM',
                            'WELL_STREAM_NUM', 'MIXTURE_NUM', 'SEP_NUM', 'MIXTURE_ITEMS_NUM', 'CON_GROUP_NUM',
                            'WELL_LIST_NUM', 'DYN_WELL_LIST_NUM'], domain=None, dtype=int),

    'VFPPDIMS': dict(attrs=['FLOW_VAL_NUM', 'TUB_HEAD_PRES_VAL_NUM', 'WFR_NUM', 'GFR_NUM', 'ALQ_NUM', 'VFP_TAB_NUM'],
                     domain=None, dtype=int),

    'VFPIDIMS': dict(attrs=['FLOW_VAL_NUM', 'THP_VAL_NUN', 'VFP_TAB_NUM'], domain=None, dtype=int),

    'AQUDIMS': dict(attrs=['AQUNUM_LINES_NUM', 'AQUCON_LINES_NUM', 'AQUTAB_LINES_NUM', 'AQUCT_INF_TABLE_ROW_NUM',
                           'AQU_ANALYTIC_NUM', 'AQU_AN_GRID_BLOCKS_NUM', 'AQU_LIST_NUM', 'AQU_IN_LIST_NUM'],
                    domain=None, dtype=int)

}


class DataTypes(Enum):
    STRING = auto()
    DATE = auto()
    VECTOR = auto()
    NUMBER = auto()
    STATEMENT_LIST = auto()
    ARRAY = auto()
    TABLE_SET = auto()
    PARAMETERS = auto()
    OBJECT_LIST = auto()

TABLE_COLUMNS = {
    'RUNCTRL': ('Parameter', 'Value'),
    'TNAVCTRL': ('Parameter', 'Value'),
    'DENSITY': ('DENSO', 'DENSW', 'DENSG'),
}

DTYPES = {
    'DIMENS': int,
    'ACTNUM': bool,
}


STATEMENT_LIST_INFO = {
    'RUNCTRL': {
        'columns': ['PARAMETER', 'VALUE'],
        'dtypes': ['text', 'text']
    },
    'WELSPECS': {
        'columns': [
            'NAME', 'GROUP', 'IW', 'JW', 'REF_DEPTH', 'PHASE', 'DRAINAGE_RADIUS', 'INFLOW_EQUATION_FLAG',
            'SHUT_OR_STOP', 'CROSFLOW_ABILITY_FLAG', 'PRESSURE_TABLE_NUMBER', 'DENSITY_CALCULATION_TYPE',
            'PRESSURE_RETRIEVING'
        ],
        'dtypes': [
            'text', 'text', 'int', 'int', 'float', 'text', 'float', 'text', 'text', 'text', 'int', 'text', 'text',
            'text'
        ]
    },
    'COMPDAT': {
        'columns': [
            'WELL', 'IW', 'JW', 'K1', 'K2', 'STATUS', 'SAT_TABLE_NUM', 'CF', 'DIAMETER', 'KH_EFF', 'SKIN',
            'D_FACTOR', 'DIRECTION', 'RADIUS_EFF'
        ],
        'dtypes': [
            'text', 'int', 'int', 'int', 'int', 'text', 'int', 'float', 'float', 'float', 'float', 'float', 'text',
            'float'
        ]
    },
    'WCONINJE': {
        'columns': [
            'WELL', 'FLUID', 'MODE', 'CONTROL', 'SURFACE_RATE', 'RESERVOIR_RATE', 'BHP', 'THP', 'VFP_TABLE_NUM',
            'OIL_GAS_CONCETRATION', '', 'SURFACE_OIL_PROPORTION', 'SURFACE_WATER_PROPORTION',
            'SURFACE_GAS_CONCETRATION'
        ],
        'dtypes': [
            'text', 'text', 'text', 'text', 'float', 'float', 'float', 'float', 'int', 'float',
            'text', 'float', 'float', 'float'
        ]
    }
}

DATA_DIRECTORY = {
    "RUNSPEC": {
        'TITLE': DataTypes.STRING,
        'MULTOUT': None,
        'MULTOUTS': None,
        'UNIFOUT': None,
        'START': DataTypes.STRING,
        'METRIC': None,
        **{fluid: None for fluid in FLUID_KEYWORDS},
        'DIMENS': DataTypes.VECTOR,
        'NUMRES': DataTypes.VECTOR,
        **{kw: DataTypes.TABLE_SET for kw in DIMS_KEYWORDS},
        'RUNCTRL': DataTypes.STATEMENT_LIST,
        'TNAVCTRL': DataTypes.STATEMENT_LIST
    },
    "GRID": {
        'MAPAXES': DataTypes.VECTOR,
        **{keyword: DataTypes.ARRAY for keyword in ORTHOGONAL_GRID_KEYWORDS},
        **{keyword: DataTypes.ARRAY for keyword in ROCK_GRID_KEYWORDS}
    },
    "PROPS": {
        **{keyword: DataTypes.TABLE_SET for keyword in TABLES_KEYWORDS}
    },
    "REGIONS": {
    },
    "SOLUTION": {
        'EQUIL': DataTypes.TABLE_SET,
        'RPTSOL': DataTypes.PARAMETERS
    },
    "SUMMARY": {
        **{keyword: None for keyword in FIELD_SUMMARY_KETWORDS},
        **{keyword: DataTypes.OBJECT_LIST for keyword in WELL_SUMMARY_KEYWORDS},
        **{keyword: None for keyword in TOTAL_SUMMARY_KEYWORDS},
        'EXCEL': None,
        'RPTONLY': None
    },
    'SCHEDULE': {
        'RPTSCHED': DataTypes.PARAMETERS,
        'RPTRST': DataTypes.PARAMETERS,
        'WELSPECS': DataTypes.STATEMENT_LIST,
        **{keyword: DataTypes.STATEMENT_LIST for keyword in SCHEDULE_KEYWORDS}
    }
}
def dump_keyword(spec, val, buf, include_path):
    _DUMP_ROUTINES[spec.data_type](spec.keyword, val, buf, include_path)
    buf.write('\n')
    return buf

def _dump_array(keyword, val, buf, include_dir):
    buf.write(keyword+'\n')
    with open(os.path.join(include_dir, f'{keyword}.inc'), 'w') as inc_buf:
        _dump_array_ascii(inc_buf, val.reshape(-1), fmt='%.3f')
    buf.write('\t'.join(('INCLUDE', f'"{os.path.join(os.path.split(include_dir)[1], f"{keyword}.inc")}"')))
    buf.write('\n/\n')

def _dump_table(keyword, val, buf):
    buf.write(keyword + '\n')
    for table in val:
        for _, row in table.iterrows():
            buf.write('\t'.join([str(v) for v in row.values] + ['\n']))
        buf.write('/\n')

_DUMP_ROUTINES = {
    DataTypes.STRING: lambda keyword, val, buf, _: buf.write('\n'.join([keyword, val, '/\n'])),
    DataTypes.STATEMENT_LIST: lambda keyword, val, buf, _: buf.write('\n'.join([keyword] +
       ['\t'.join([str(value) for value in row[1].values.tolist() + ['/']]) for row in val.iterrows()] +
        ['/\n']
)),
    DataTypes.VECTOR: lambda keyword, val, buf, _: buf.write('\n'.join([keyword, '\t'.join(map(str, val)), '/\n'])),
    DataTypes.ARRAY: _dump_array,
    DataTypes.TABLE_SET: lambda keyword, val, buf, _: _dump_table(keyword, val, buf),
    None: lambda keyword, _, buf, ___: buf.write(f'{keyword}\n')
}


def _dump_array_ascii(buffer, array, header=None, fmt='%f', compressed=True):
    """Writes array-like data into an ASCII buffer.

    Parameters
    ----------
    buffer : buffer-like
    array : 1d, array-like
        Array to be saved
    header : str, optional
        String to be written line before the array
    fmt : str or sequence of strs, optional
        Format to be passed into ``numpy.savetxt`` function. Default to '%f'.
    compressed : bool
        If True, uses compressed typing style
    """
    if header is not None:
        buffer.write(header + '\n')

    if compressed:
        i = 0
        items_written = 0
        while i < len(array):
            count = 1
            while (i + count < len(array)) and (array[i + count] == array[i]):
                count += 1
            if count <= 4:
                buffer.write(' '.join([fmt % array[i]] * count))
                items_written += count
            else:
                buffer.write(str(count) + '*' + fmt % array[i])
                items_written += 1
            i += count
            if items_written > MAX_STRLEN:
                buffer.write('\n')
                items_written = 0
            else:
                buffer.write(' ')
        buffer.write('\n')
    else:
        for i in range(0, len(array), MAX_STRLEN):
            buffer.write(' '.join([fmt % d for d in array[i:i + MAX_STRLEN]]))
            buffer.write('\n')
        buffer.write('\n')
