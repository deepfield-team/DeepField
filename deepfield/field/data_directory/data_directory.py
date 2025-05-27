from collections.abc import Sequence
from enum import Enum, auto
import os
from typing import Any, Callable, NamedTuple, Optional

import pandas as pd
import numpy as np
from pandas.errors import SpecificationError


INT_NAN = -99999999

FLUID_KEYWORDS = ('OIL', 'GAS', 'WATER', 'DISGAS')
ORTHOGONAL_GRID_KEYWORDS = ('DX', 'DY', 'DZ', 'TOPS', 'ACTNUM')
ROCK_GRID_KEYWORDS = ('PORO', 'PERMX', 'PERMY', 'PERMZ', 'NTG')
TABLES_KEYWORDS = ('DENSITY', 'PVCDO', 'PVTW', 'ROCK', 'SWOF')
FIELD_SUMMARY_KEYWORDS = ('FOPR', 'FWPR', 'FWIR', 'FHPV', 'FMWPR', 'FMWPT', 'FMWPA',
                          'FGIP', 'FGOR', 'FGORH', 'FGIP', 'FGPR', 'FGPRH',
                          'FGPT', 'FGPTH', 'FLPR', 'FLPRH', 'FLPT', 'FLPTH', 'FOIP', 'FOPRH',
                          'FOPTH', 'FPR', 'FWCT', 'FWCTH', 'FWIRH',
                          'FWITH', 'FWPRH', 'FWPTH', 'FVPR', 'FVPT', 'FTPRAQW', 'FTPRFW', 'FTPRMW',
                          'FTPRFO', 'FTPRMO')
WELL_SUMMARY_KEYWORDS = ('WOPR', 'WWPR', 'WWIR', 'WLPR', 'WBHP', 'WBP9', 'WBP', 'WGIR', 'WGIRH',
                         'WGIT', 'WGITH', 'WGOR', 'WGORH', 'WGPR', 'WGPRH', 'WGPT', 'WGPTH', 'WLPRH',
                         'WLPT', 'WLPTH', 'WOPRH', 'WOPTH', 'WWCT', 'WWCTH', 'WWIRH', 'WWIT', 'WWITH',
                         'WWPRH', 'WWPT', 'WWPTH', 'WTPRAQW', 'WTRFW', 'WTPRMV', 'WPIW', 'WPIO', 'WPIG',
                         'WWPP', 'WOPP', 'WGPP', 'WOPT', 'WTPRFW', 'WTPRMW',)
REGION_SUMMARY_KEYWORDS = ('RGIPL', 'ROIP', 'RWIP','RGIP', 'RPR', 'ROE')
TOTAL_SUMMARY_KEYWORDS = ('FOPT', 'FWPT', 'FWIT')
SCHEDULE_KEYWORDS = ('WELSPECS','COMPDAT', 'WCONPROD', 'WCONINJE')
DIMS_KEYWORDS = ('TABDIMS', 'EQLDIMS', 'REGDIMS', 'WELLDIMS', 'VFPPDIMS', 'VFPIDIMS',
                 'AQUDIMS')
MODEL_SUMMARY_KEYWORDS = ('TIMESTEP', 'ELAPSED', 'TCPU', 'MLINEARS', 'MSUMLINS', 'MSUMNEWT', 'NEWTON',
                          'MLINEARS', 'MSUMLINS', 'MSUMNEWT', 'NEWTON', 'NLINEARS', 'STEPTYPE')
GROUP_SUMMARY_KEYWORDS = ('GTPRAQW', 'GTPRFW', 'GTPRMW', 'GGOR', 'GGPR', 'GGPRH', 'GGPT', 'GGPTH',
                          'GLPR', 'GLPRH', 'GLPT', 'GLPTH', 'GOPR', 'GOPRH', 'GOPT', 'GOPTH', 'GWCT',
                          'GWCTH', 'GWPR', 'GWPRH', 'GWPT', 'GWPTH', 'GGORH')
REGIONS_SUMMARY_KEYWORDS = ()

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
                         'RVVD_PDVD_TABLE', 'ACCURACY'], domain=None),
}


class DataTypes(Enum):
    STRING = auto()
    SINGLE_STATEMENT = auto()
    STATEMENT_LIST = auto()
    ARRAY = auto()
    TABLE_SET = auto()
    PARAMETERS = auto()
    OBJECT_LIST = auto()
    RECORDS = auto()

class SECTIONS(Enum):
    RUNSPEC = 'RUNSPEC'
    NONE = ''
    GRID = 'GRID'
    SCHEDULE = 'SCHEDULE'
    EDIT = 'EDIT'
    PROPS = 'PROPS'
    SOLUTION = 'SOLUTION'
    SUMMARY = 'SUMMARY'
    REGIONS = 'REGIONS'

TABLE_COLUMNS = {
    'RUNCTRL': ('Parameter', 'Value'),
    'TNAVCTRL': ('Parameter', 'Value'),
    'DENSITY': ('DENSO', 'DENSW', 'DENSG'),
}

DTYPES = {
    'ACTNUM': bool,
    'TSTEP': int,
}

class TableSpecification(NamedTuple):
    columns: Sequence[str]
    domain: Sequence[int] | None
    dtypes: Sequence[str] | str = 'float'

class ParametersSpecification(NamedTuple):
    tabulated: bool=False

class StatementSpecification(NamedTuple):
    columns: Sequence[str]
    dtypes: Sequence[str]
    terminated: bool=True

class RecordsSpecification(NamedTuple):
    specifications: Sequence[StatementSpecification]

class ArraySpecification(NamedTuple):
    dtype: type

class ObjectSpecification(NamedTuple):
    terminated: bool=False

class KeywordSpecification(NamedTuple):
    keyword: str
    type: DataTypes | None
    specification: (StatementSpecification |
        RecordsSpecification | ObjectSpecification |
        None | ArraySpecification | TableSpecification | ParametersSpecification)
    sections: Sequence[SECTIONS]


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
    },
    'WCONPROD': {
        'columns': [ 'WELL', 'MODE', 'CONTROL', 'OIL_RATE', 'WATER_RATE', 'GAS_RATE', 'SURFACE_LIQUID_RATE',
            'RESERVOIR_LIQUID_RATE', 'BHP', 'THP', 'VFP_TABLE_NUM', 'ALQ', 'WET_GAS_PRODUCTION_RATE',
            'TOTAL_MOLAR_RATE', 'STEAM_PRODUCTION', 'PRESSURE_OFFSET', 'TEMPERATURE_OFFSET',
            'CALORIFIC_RATE', 'LINEARLY_COMBINED_RATE_TARGET', 'NGL_RATE'
        ],
        'dtypes': ['text'] * 3 + ['float'] * 7 + ['int'] + ['float'] * 9
    },
    'DIMENS': {
        'columns': [
            'NX', 'NY', 'NZ'
        ],
        'dtypes': ['int', 'int', 'int'],
    },
    'NUMRES': {
        'columns': ['NRES'],
        'dtypes': ['int']
    },
    'NSTACK': {
        'columns': ['NSTACK'],
        'dtypes': ['int']
    },
    'MAPAXES': {
        'columns': ['X1', 'Y1', 'X0', 'Y0', 'X2', 'Y2'],
        'dtypes': ['float'] * 6,
    },

    'TABDIMS': {'columns': ['SAT_REGIONS_NUM', 'PVT_REGIONS_NUM', 'SAT_NODES_NUM', 'PVT_NODES_NUM',
                           'FIP_REGIONS_NUM', 'OIL_VAP_NODES_NUM', 'OGR_NODES_NUM', 'SAT_END_POINT_NUM',
                           'EOS_REGIONS_NUM', 'EOS_SURFACE_REGIONS_NUM', 'FLUX_REGIONS_NUM', 'THERMAL_REGIONS_NUM',
                           'ROCK_TABLES_NUM', 'PRESSURE_MAINTAINACE_REGIONS_NUM', 'TEMPERATURE_NODES_NUM',
                           'TRANSPORT_COEFFICIENTS_NUM'],
                'dtypes': ['int'] * 16},

    'EQLDIMS': {'columns': ['EQL_NUM', 'EQL_NODE_NUM', 'DEPTH_NODE_MAX_NUM', 'INIT_TRAC_CONC_NUM',
                           'INIT_TRAC_CONC_NODE_NUM'],
                'dtypes': ['int'] * 5},

    'REGDIMS': {'columns': ['FIP_REGIONS_NUM', 'FIP_FAMILIES_NUM', 'RESERVOIR_REGIONS_NUM',
                           'FLUX_REGIONS_NUM', 'TRACK_REGIONS_NUM', 'COAL_REG_NUM', 'OPER_REGIONS_NUM',
                           'WORK_NUM', 'IWORK_NUM', 'PLMIX_REGIONS_NUM'],
                'dtypes': ['int'] * 10},

    'WELLDIMS': {'columns': ['WELL_NUM', 'CONN_NUM', 'GROUP_NUM', 'WELL_IN_GROUP_NUM', 'SEP_STAGES_NUM',
                            'WELL_STREAM_NUM', 'MIXTURE_NUM', 'SEP_NUM', 'MIXTURE_ITEMS_NUM', 'CON_GROUP_NUM',
                            'WELL_LIST_NUM', 'DYN_WELL_LIST_NUM'],
                 'dtypes': ['int'] * 12},

    'VFPPDIMS': {'columns': ['FLOW_VAL_NUM', 'TUB_HEAD_PRES_VAL_NUM', 'WFR_NUM', 'GFR_NUM', 'ALQ_NUM', 'VFP_TAB_NUM'],
                 'dtypes': ['int'] * 6},

    'VFPIDIMS': {'columns': ['FLOW_VAL_NUM', 'THP_VAL_NUN', 'VFP_TAB_NUM'],
                 'dtypes': ['int']*3},

    'AQUDIMS': {'columns': ['AQUNUM_LINES_NUM', 'AQUCON_LINES_NUM', 'AQUTAB_LINES_NUM', 'AQUCT_INF_TABLE_ROW_NUM',
                           'AQU_ANALYTIC_NUM', 'AQU_AN_GRID_BLOCKS_NUM', 'AQU_LIST_NUM', 'AQU_IN_LIST_NUM'],
                'dtypes': ['int'] * 8},

    'SPECGRID': {'columns': ['NX', 'NY', 'NZ', 'NRES', 'COORDINATE_SYSTEM'],
                 'dtypes': ['int'] * 4 + ['text']},

    'COPY': {'columns': ['SOURCE', 'DEST', 'IMIN', 'IMAX', 'JMIN', 'JMAX', 'KMIN', 'KMAX'],
             'dtypes': ['text'] * 2 + ['int'] * 6},

    'MULTIPLY': {
        'columns': ['ARR', 'MULTIPLYER', 'IMIN', 'IMAX', 'JMIN', 'JMAX', 'KMIN', 'KMAX'],
        'dtypes': ['text', 'float'] + ['int'] * 6
    }
}

RECORDS_INFO = {
    'TUNING': [
        {
            'columns': ['TSINIT', 'TSMAXZ', 'TSMINZ', 'TSMCHP', 'TSFMAX', 'TSFMIN', 'TSFCNV', 'TFDIFF', 'TFRUPT',
                        'TMAXWC'],
            'dtypes': ['float']*10
        },
        {
            'columns': ['TRGTTE', 'TRGCNV', 'TRGMBE', 'TRGLCV', 'XXXTTE', 'XXXCNV', 'XXXMBE', 'XXXLCV', 'XXXWFL',
                        'TRGFIP', 'TRGSFT', 'THIONX', 'TRWGHT'],
            'dtypes': ['float']*12 + ['int']
        },
        {
            'columns': ['NEWTMX', 'NEWTMN', 'LITMAX', 'LITMIN', 'MXWSIT', 'MXWPIT', 'DDPLIM', 'DDSLIM', 'TRGDPR',
                        'XXXDPR'],
            'dtypes': ['int']*6 + ['float'] * 4,
        }
    ]
}

DATA_DIRECTORY = {
    '': {},
    "RUNSPEC": {
        'TITLE': DataTypes.STRING,
        'MULTOUT': None,
        'MULTOUTS': None,
        'UNIFOUT': None,
        'START': DataTypes.STRING,
        'METRIC': None,
        **{fluid: None for fluid in FLUID_KEYWORDS},
        'DIMENS': DataTypes.SINGLE_STATEMENT,
        'NUMRES': DataTypes.SINGLE_STATEMENT,
        **{kw: DataTypes.SINGLE_STATEMENT for kw in DIMS_KEYWORDS},
        'NSTACK': DataTypes.SINGLE_STATEMENT,
        'RUNCTRL': DataTypes.STATEMENT_LIST,
        'TNAVCTRL': DataTypes.STATEMENT_LIST,
        'TUNING': DataTypes.RECORDS
    },
    "GRID": {
        'SPECGRID': DataTypes.SINGLE_STATEMENT,
        'INIT': None,
        'MAPAXES': DataTypes.SINGLE_STATEMENT,
        **{keyword: DataTypes.ARRAY for keyword in ORTHOGONAL_GRID_KEYWORDS},
        **{keyword: DataTypes.ARRAY for keyword in ROCK_GRID_KEYWORDS},
        'MULTIPLY': DataTypes.STATEMENT_LIST,
        'COPY': DataTypes.STATEMENT_LIST
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
        **{keyword: None for keyword in FIELD_SUMMARY_KEYWORDS},
        **{keyword: DataTypes.OBJECT_LIST for keyword in WELL_SUMMARY_KEYWORDS},
        **{keyword: None for keyword in TOTAL_SUMMARY_KEYWORDS},
        'EXCEL': None,
        'RPTONLY': None
    },
    'SCHEDULE': {
        'RPTSCHED': DataTypes.PARAMETERS,
        'RPTRST': DataTypes.PARAMETERS,
        'WELSPECS': DataTypes.STATEMENT_LIST,
        'TSTEP': DataTypes.ARRAY,
        **{keyword: DataTypes.STATEMENT_LIST for keyword in SCHEDULE_KEYWORDS},
        'TUNING': DataTypes.RECORDS
    }
}


DATA_DIRECTORY = {
    'TITLE': KeywordSpecification('TITLE', DataTypes.STRING, None, (SECTIONS.RUNSPEC,)),
    **{kw: KeywordSpecification(kw, None, None, (SECTIONS.RUNSPEC,)) for kw in [
        'MULTOUT', 'MULTOUTS', 'UNIFOUT', 'METRIC'
    ]},
    'START': KeywordSpecification('START', DataTypes.STRING, None, (SECTIONS.RUNSPEC,)),
    **{kw: KeywordSpecification(kw, None, None, (SECTIONS.RUNSPEC,)) for kw in FLUID_KEYWORDS},
    **{kw: KeywordSpecification(kw, DataTypes.SINGLE_STATEMENT, StatementSpecification(
        STATEMENT_LIST_INFO[kw]['columns'], STATEMENT_LIST_INFO[kw]['dtypes']),
                                (SECTIONS.RUNSPEC,)) for kw in ['DIMENS', 'NUMRES', 'NSTACK'] + list(DIMS_KEYWORDS)
    },
    **{kw: KeywordSpecification(kw, DataTypes.STATEMENT_LIST, StatementSpecification(
        STATEMENT_LIST_INFO[kw]['columns'], STATEMENT_LIST_INFO[kw]['dtypes']
    ), (SECTIONS.RUNSPEC,)) for kw in ('RUNCTRL',)},
    'TUNING': KeywordSpecification('TUNING', DataTypes.RECORDS, RecordsSpecification([StatementSpecification(
        r['columns'], r['dtypes']) for r in RECORDS_INFO['TUNING']]), (SECTIONS.RUNSPEC, SECTIONS.SCHEDULE)),
    'SPECGRID': KeywordSpecification(
        'SPECGRID', DataTypes.SINGLE_STATEMENT, StatementSpecification(
            STATEMENT_LIST_INFO['SPECGRID']['columns'], STATEMENT_LIST_INFO['SPECGRID']['dtypes']
        ), (SECTIONS.GRID,)
    ),
    'INIT': KeywordSpecification('INIT', None, None, (SECTIONS.GRID,)),
    'MAPAXES': KeywordSpecification('MAPAXES', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        STATEMENT_LIST_INFO['MAPAXES']['columns'], STATEMENT_LIST_INFO['MAPAXES']['dtypes']
    ), (SECTIONS.GRID,)),
    **{kw: KeywordSpecification(kw, DataTypes.ARRAY,
                                ArraySpecification(DTYPES[kw] if kw in DTYPES else float),
                                (SECTIONS.GRID,)) for kw in list(ORTHOGONAL_GRID_KEYWORDS) + list(ROCK_GRID_KEYWORDS)},
    'MULTIPLY': KeywordSpecification('MULTIPLY', DataTypes.STATEMENT_LIST, StatementSpecification(
        STATEMENT_LIST_INFO['MULTIPLY']['columns'], STATEMENT_LIST_INFO['MULTIPLY']['dtypes']
    ), (SECTIONS.GRID,) ),
    'COPY': KeywordSpecification('COPY', DataTypes.STATEMENT_LIST, StatementSpecification(
        STATEMENT_LIST_INFO['COPY']['columns'], STATEMENT_LIST_INFO['COPY']['dtypes']
    ), (SECTIONS.GRID,) ),
    **{kw: KeywordSpecification(kw, DataTypes.TABLE_SET, TableSpecification(
        TABLE_INFO[kw]['attrs'], TABLE_INFO[kw]['domain']
    ), (SECTIONS.PROPS,)) for kw in TABLES_KEYWORDS},
    'EQUIL': KeywordSpecification('EQUIL', DataTypes.TABLE_SET, TableSpecification(
        TABLE_INFO['EQUIL']['attrs'],
        TABLE_INFO['EQUIL']['domain'],
        ['float'] * 6 + ['int'] * 3
    ), (SECTIONS.SOLUTION,)),
    'RPTSOL': KeywordSpecification('RPTSOL', DataTypes.PARAMETERS, ParametersSpecification(), (SECTIONS.SOLUTION,)),
    **{kw: KeywordSpecification(kw, None, None, (SECTIONS.SUMMARY,)) for kw in FIELD_SUMMARY_KEYWORDS},
    **{kw: KeywordSpecification(kw, DataTypes.OBJECT_LIST, None, (SECTIONS.SUMMARY,)) for kw in WELL_SUMMARY_KEYWORDS},
    **{kw: KeywordSpecification(kw, None, None, (SECTIONS.SUMMARY,)) for kw in TOTAL_SUMMARY_KEYWORDS},
    'EXCEL': KeywordSpecification('EXCEL', None, None, (SECTIONS.SUMMARY,)),
    'RPTONLY': KeywordSpecification('RPTONLY', None, None, (SECTIONS.SUMMARY,)),
    'RPTSCHED': KeywordSpecification('RPTSCHED', DataTypes.PARAMETERS, ParametersSpecification(), (SECTIONS.SCHEDULE,)),
    'RPTRST': KeywordSpecification('RPTRST', DataTypes.PARAMETERS, ParametersSpecification(), (SECTIONS.SCHEDULE,)),
    'WELSPECS': KeywordSpecification('WELSPECS', DataTypes.STATEMENT_LIST, None, (SECTIONS.SCHEDULE,)),
    'TSTEP': KeywordSpecification('TSTEP', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.SCHEDULE,)),
    **{kw: KeywordSpecification(kw, DataTypes.STATEMENT_LIST, StatementSpecification(
        STATEMENT_LIST_INFO[kw]['columns'], STATEMENT_LIST_INFO[kw]['dtypes'],
    ), (SECTIONS.SCHEDULE,)) for kw in SCHEDULE_KEYWORDS},
    **{kw: KeywordSpecification(kw, None, None, [val for val in SECTIONS]) for kw in ('NOECHO', 'ECHO', 'END')},
    'INCLUDE': KeywordSpecification('INCLUDE', DataTypes.STRING, None, [val for val in SECTIONS]),
    'REPORTSCREEN': KeywordSpecification('REPORTSCREEN', DataTypes.PARAMETERS, ParametersSpecification(
        tabulated=True,
    ), [val for val in SECTIONS]),
    'ENDSCALE': KeywordSpecification('ENDSCALE', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['DIRECTIONAL_SWITCH', 'IRREVERSIBLE_SWITCH', 'SAT_POINTS_NUM', 'NODE_NUM', 'COMBININNG',
         'EQUILIBRATION'],
        ['text'] * 2 + ['int'] * 3 + ['text']), (SECTIONS.RUNSPEC,)),
    'FAULTDIM' : KeywordSpecification('FAULTDIM', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['N'], ['int']
    ), (SECTIONS.RUNSPEC,)),
    'EQLOPTS': KeywordSpecification('EQLOPTS', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['OPTION'], ['text']
    ), (SECTIONS.RUNSPEC,)),
    'MESSAGES': KeywordSpecification('MESSAGES', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        [f'PRINT_LIM_SEV{i}' for i in range(1, 7)] + [f'STOP_LIM_SEV{i}' for i in range(1, 7)],
        ['int']*12
    ), [val for val in SECTIONS]),
    'GRIDFILE': KeywordSpecification('GRIDFILE', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['_', 'DUMP_INIT_EGRID'], ['int']*2
    ), (SECTIONS.GRID,)),
    'MINPV': KeywordSpecification('MINPV', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['VAL'], ['float']
    ), (SECTIONS.GRID,)),
    'PINCH': KeywordSpecification('PINCH', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['THRESHOLD_THICKNESS', 'MINPV_NN_CONTROL', 'MAX_GAP', 'NN_TRANSMISSIBILITY_METHOD',
         'VERT_TRANSMISSIBILITY_METHOD'], ['float', 'text', 'float', 'text', 'text'],
    ), (SECTIONS.GRID,)),
    'FAULTS': KeywordSpecification('FAULTS', DataTypes.STATEMENT_LIST, StatementSpecification(
        ['NAME', 'I1', 'I2', 'J1', 'J2', 'K1', 'K2', 'FACE'], ['text'] + ['int']*6 + ['text']
    ), (SECTIONS.GRID,)),
    'JFUNC': KeywordSpecification('JFUNC',  DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['PHASE', 'STW', 'STG', 'ALPHA', 'BETA', 'PERM_DIR'],
        ['text'] + ['float'] * 4 + ['text']
    ), (SECTIONS.GRID,)),
    'COORD': KeywordSpecification('COORD', DataTypes.ARRAY, ArraySpecification(
       float
    ), (SECTIONS.GRID,)),
    'ZCORN': KeywordSpecification('ZCORN', DataTypes.ARRAY, ArraySpecification(
       float
    ), (SECTIONS.GRID,)),
    'EDITNNC': KeywordSpecification('EDITNNC', DataTypes.STATEMENT_LIST, StatementSpecification(
        ['I1', 'J1', 'K1', 'I2', 'J2', 'K2', 'TRANSMISSIBILITY_MULT', 'SAT_REG1', 'SAT_REG2',
         'PVT_REG1', 'PVT_REG2', 'DIR1', 'DIR2', 'DIFFUSIVITY_MULT'],
        ['int'] * 6 + ['float'] + ['int'] * 4 + ['text'] * 2 + ['float']
    ), (SECTIONS.EDIT,)),
    'STONE1': KeywordSpecification('STONE1', None, None, (SECTIONS.PROPS,)),
    'STONE2': KeywordSpecification('STONE2', None, None, (SECTIONS.PROPS,)),
    'PVTO': KeywordSpecification('PVTO', DataTypes.TABLE_SET, TableSpecification(TABLE_INFO['PVTO']['attrs'],
                                                                                 TABLE_INFO['PVTO']['domain']),
                                 (SECTIONS.PROPS,)),
    'PVDG': KeywordSpecification('PVDG', DataTypes.TABLE_SET, TableSpecification(TABLE_INFO['PVDG']['attrs'],
                                                                                 TABLE_INFO['PVDG']['domain']),
                                 (SECTIONS.PROPS,)),
    'SGOF': KeywordSpecification('SGOF', DataTypes.TABLE_SET, TableSpecification(TABLE_INFO['SGOF']['attrs'],
                                                                                 TABLE_INFO['SGOF']['domain']),
                                 (SECTIONS.PROPS,)),
    'SCALECRS': KeywordSpecification('SCALECRS', DataTypes.SINGLE_STATEMENT, StatementSpecification(
        ['VALUE'], ['text']
    ), (SECTIONS.PROPS,)),
    'PVTNUM': KeywordSpecification('PVTNUM', DataTypes.ARRAY, ArraySpecification(int),
                                   (SECTIONS.REGIONS, SECTIONS.SCHEDULE)),
    'EQLNUM': KeywordSpecification('EQLNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)),
    'FIPFAULT': KeywordSpecification('FIPFAULT', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)),
    'SATNUM': KeywordSpecification('SATNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)),
    'FIPNUM': KeywordSpecification('FIPNUM', DataTypes.ARRAY, ArraySpecification(int), (SECTIONS.REGIONS,)),
    'PBVD': KeywordSpecification('PBVD', DataTypes.TABLE_SET, TableSpecification(
        ['DEPTH', 'PB'], [0]
    ), (SECTIONS.SOLUTION,)),
    'THPRES': KeywordSpecification('THPRES', DataTypes.STATEMENT_LIST, StatementSpecification(
        ['REG1', 'REG2', 'TH'], ['int', 'int', 'float']
    ), (SECTIONS.SOLUTION,),),
    'OUTSOL': KeywordSpecification('OUTSOL', DataTypes.PARAMETERS, ParametersSpecification(),
                                   (SECTIONS.SOLUTION, SECTIONS.SCHEDULE)),
    **{kw: KeywordSpecification(kw, None, None, [val for val in SECTIONS]) for kw in ('SKIP', 'SKIPOFF', 'SKIPON',
                                                                                      'ENDSKIP')},
    **{kw: KeywordSpecification(kw, DataTypes.OBJECT_LIST, None,
                                (SECTIONS.SUMMARY,)) for kw in REGION_SUMMARY_KEYWORDS},
    **{kw: KeywordSpecification(kw, None, None, (SECTIONS.SUMMARY,)) for kw in MODEL_SUMMARY_KEYWORDS},
    **{kw: KeywordSpecification(kw, DataTypes.OBJECT_LIST, None,
                                (SECTIONS.SUMMARY,)) for kw in GROUP_SUMMARY_KEYWORDS},
    'WCONHIST': KeywordSpecification('WCONHIST', DataTypes.STATEMENT_LIST, StatementSpecification(
        [ 'WELL', 'MODE', 'CONTROL', 'OIL_RATE', 'WATER_RATE', 'GAS_RATE',
         'VFP_TABLE_NUM', 'ALQ', 'THP', 'BHP', 'WET_GAS_PRODUCTION_RATE',
         'NGL_RATE'],
        ['text'] * 3 + ['float'] * 3 + ['int'] + ['float'] * 5

    ), (SECTIONS.SCHEDULE,)),
    'DATES': KeywordSpecification('DATES', DataTypes.OBJECT_LIST, ObjectSpecification(terminated=True),
                                  (SECTIONS.SCHEDULE,)),
    'WCONINJH': KeywordSpecification('WCONINJH', DataTypes.STATEMENT_LIST, StatementSpecification(
        ['WELL', 'FLUID', 'MODE', 'SURFACE_RATE', 'BHP', 'THP', 'VFP_TABLE_NUM', 'OIL_GAS_CONCETRATION',
         'SURFACE_OIL_PROPORTION', 'SURFACE_WATER_PROPORTION', 'SURFACE_GAS_CONCETRATION', 'CONTROL'],
        ['text', 'text', 'text', 'float', 'float', 'float', 'int', 'float', 'float', 'float', 'float',
         'text']
    ), (SECTIONS.SCHEDULE,)),
    'RPTRSTD': KeywordSpecification('RPTRSTD', DataTypes.OBJECT_LIST, ObjectSpecification(), (SECTIONS.SOLUTION,))
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
