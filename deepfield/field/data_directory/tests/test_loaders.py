import itertools
from typing import NamedTuple
import pytest
import numpy as np
import pandas as pd
from deepfield.field.data_directory.load_utils import LOADERS, decompress_array, parse_vals

from deepfield.field.data_directory.data_directory import ArrayWithUnits, DataTypes, DATA_DIRECTORY
from deepfield.field.parse_utils.ascii import INT_NAN

TEST_DATA = {
    DataTypes.STRING: [
        (
            '\n'.join(('TITLE', 'abc /', '', '')),
            ('TITLE', 'abc')
        ),
        (
            '\n'.join(('TITLE', 'abc', '/', '')),
            ('TITLE', 'abc')
        ),
        (
            '\n'.join(('TITLE', 'abc', '', '')),
            ('TITLE', 'abc')
        ),
        (
            '\n'.join((
                'TITLE',
                'MODEL COMP FAULT TEST',
                '',
                'abc'
            )),
            ('TITLE', 'MODEL COMP FAULT TEST')
        ),
        (
            '\n'.join(('INCLUDE', '"abc/abc"', '/')),
            ('INCLUDE', 'abc/abc')
        ),
        (
            '\n'.join(('INCLUDE', '"abc/abc" /')),
            ('INCLUDE', 'abc/abc')
        ),
        (
            '\n'.join(('INCLUDE', "'abc/abc'", '/')),
            ('INCLUDE', 'abc/abc')
        ),
        (
            '\n'.join(('INCLUDE', "a'abc/abc'", '/')),
            ('INCLUDE', "aabc/abc")
        ),
        (
            '\n'.join(('INCLUDE', "'abc/abc'a", '/')),
            ('INCLUDE', "abc/abca")
        ),
        (
            '\n'.join(('START', '01 JUL 1984 /', '/')),
            ('START', (pd.to_datetime('1984-07-01')))
        ),
        (
            '\n'.join((
                'START',
                "01 'JUN' 2010 /"
            )),
            ('START', pd.to_datetime('2010-06-01'))
        )
    ],
    DataTypes.TABLE_SET: [
        (
            '\n'.join((
                'EQUIL',
                '2300 200 2500 0.1 2300 0.001 /',
                '2310 205 2520 0.05 2310 0.0 /'
                '2305 210 2510 1* 2305 1* /'
            )),
            (
                'EQUIL',
                (
                    pd.DataFrame([
                        [2300.0, 200.0, 2500.0, 0.1, 2300.0, 0.001, INT_NAN, INT_NAN, INT_NAN],
                    ], columns=DATA_DIRECTORY['EQUIL'].specification.columns),
                    pd.DataFrame([
                        [2310.0, 205.0, 2520.0, 0.05, 2310.0, 0.0, INT_NAN, INT_NAN, INT_NAN],
                    ], columns=DATA_DIRECTORY['EQUIL'].specification.columns),
                    pd.DataFrame([
                        [2305.0, 210.0, 2510.0, np.NaN, 2305.0, np.NaN, INT_NAN, INT_NAN, INT_NAN],
                    ], columns=DATA_DIRECTORY['EQUIL'].specification.columns)
                )
            )
        ),
        (
            '\n'.join((
                'EQUIL',
                '1450 141 1475 0 638 0 1 1* 10 /',
                '1450 141 1475 0 965 0 1 1* 10 /',
                '/'
            )),
            (
                'EQUIL',
                (
                    pd.DataFrame([
                        [1450.0, 141.0, 1475.0, 0.0, 638.0, 0.0, 1, INT_NAN, 10]
                    ], columns=DATA_DIRECTORY['EQUIL'].specification.columns),
                    pd.DataFrame([
                        [1450.0, 141.0, 1475.0, 0.0, 965.0, 0.0, 1, INT_NAN, 10]
                    ], columns=DATA_DIRECTORY['EQUIL'].specification.columns)
                )
            ),
        ),
        (
            '\n'.join((
                'SWOF',
                '0.42 0 0.737 0',
                '0.48728 0.000225 0.610213 0',
                '0.55456 0.00438 0.310527 0',
                '0.62184 0.023012 0.072027 0',
                '0.68912 0.069122 0.003178 0',
                '0.7564 0.151 0 0',
                '0.82368 0.267672 0 0',
                '0.89096 0.408671 0 0',
                '0.95824 0.557237 0 0',
                '1 0.645099 0 0',
                '/',
                '0 0 1 0',
                '0.3 0.002 0.81 0',
                '0.4 0.018 0.49 0',
                '0.5 0.05 0.25 0',
                '0.6 0.098 0.09 0',
                '0.7 0.162 0.01 0',
                '1 0.2 0 0',
                '/'
            )),
            (
                'SWOF',
                (
                    pd.DataFrame(np.array([
                        [0.42, 0, 0.737, 0],
                        [0.48728, 0.000225, 0.610213, 0],
                        [0.55456, 0.00438, 0.310527, 0],
                        [0.62184, 0.023012, 0.072027, 0],
                        [0.68912, 0.069122, 0.003178, 0],
                        [0.7564, 0.151, 0, 0],
                        [0.82368, 0.267672, 0, 0],
                        [0.89096, 0.408671, 0, 0],
                        [0.95824, 0.557237, 0, 0],
                        [1, 0.645099, 0, 0],
                    ]
                    ), columns=DATA_DIRECTORY['SWOF'].specification.columns).set_index(
                        DATA_DIRECTORY['SWOF'].specification.columns[DATA_DIRECTORY['SWOF'].specification.domain[0]]
                    ),
                    pd.DataFrame(np.array([
                        [0, 0, 1, 0],
                        [0.3, 0.002, 0.81, 0],
                        [0.4, 0.018, 0.49, 0],
                        [0.5, 0.05, 0.25, 0],
                        [0.6, 0.098, 0.09, 0],
                        [0.7, 0.162, 0.01, 0],
                        [1, 0.2, 0, 0],
                    ]
                    ), columns=DATA_DIRECTORY['SWOF'].specification.columns).set_index(
                        DATA_DIRECTORY['SWOF'].specification.columns[DATA_DIRECTORY['SWOF'].specification.domain[0]]
                    )
                )
            )
        ),
        (
            '\n'.join((
                'PVTO',
                '1 5 1.031 5.81 /',
                '12.33 52 1.080 5.03 /',
                '21.65 73 1.1021 4.23',
                '\t204 1.092 4.62',
                '\t321 1.016 6.02 /',
                '/',
                '1 5 1.0002 3.58 /',
                '14.87 58 1.086 2.93 /',
                '27.7 90 1.113 2.25',
                '\t234 1.1 2.88',
                '\t387 1.121 3.96 /',
                '/',
                '1 5 1.0002 3.58 /',
                '18.67 57 1.0730 2.89 /',
                '31.65 88 1.1083 2.2',
                '\t248 1.093 2.57',
                '\t334 1.073 4.23 /',
                '/'
            )),
            (
                'PVTO',
                (
                    pd.DataFrame(np.array([
                        [1, 5, 1.031, 5.81],
                        [12.33, 52, 1.08, 5.03],
                        [21.65, 73, 1.1021, 4.23],
                        [21.65, 204, 1.092, 4.62],
                        [21.65, 321, 1.016, 6.02]
                    ]), columns=DATA_DIRECTORY['PVTO'].specification.columns,).set_index(
                        [DATA_DIRECTORY['PVTO'].specification.columns[i] for
                            i in DATA_DIRECTORY['PVTO'].specification.domain]),
                    pd.DataFrame(np.array([
                        [1, 5, 1.0002, 3.58],
                        [14.87, 58, 1.086, 2.93],
                        [27.7, 90, 1.113, 2.25],
                        [27.7, 234, 1.1, 2.88],
                        [27.7, 387, 1.121, 3.96]
                    ]), columns=DATA_DIRECTORY['PVTO'].specification.columns,).set_index(
                        [DATA_DIRECTORY['PVTO'].specification.columns[i] for
                            i in DATA_DIRECTORY['PVTO'].specification.domain]),
                    pd.DataFrame(np.array([
                        [1, 5, 1.0002, 3.58],
                        [18.67, 57, 1.0730, 2.89],
                        [31.65, 88, 1.1083, 2.2],
                        [31.65, 248, 1.093, 2.57],
                        [31.65, 334, 1.073, 4.23]
                    ]), columns=DATA_DIRECTORY['PVTO'].specification.columns,).set_index(
                        [DATA_DIRECTORY['PVTO'].specification.columns[i] for
                            i in DATA_DIRECTORY['PVTO'].specification.domain]),
                )
            )
        )
    ],
    DataTypes.SINGLE_STATEMENT: [
        (
            '\n'.join((
                'TABDIMS',
                '2 4 2* 3',
                '/'
            )),
            (
                'TABDIMS',
                pd.DataFrame(
                    np.array([
                        [2, 4] + 2*[INT_NAN] + [3] + 11*[INT_NAN]
                    ]),
                    columns=DATA_DIRECTORY['TABDIMS'].specification.columns
                )
            )
        ),
        (
            '\n'.join((
                'TABDIMS',
                '2 4 2* 3/',
            )),
            (
                'TABDIMS',
                pd.DataFrame(
                    np.array([
                        [2, 4] + 2*[INT_NAN] + [3] + 11*[INT_NAN]
                    ]),
                    columns=DATA_DIRECTORY['TABDIMS'].specification.columns
                )
            )
        ),
        (
            '\n'.join((
                'ENDSCALE',
                '/'
            )),
            (
                'ENDSCALE',
                pd.DataFrame(
                    [[None] * 2 + [INT_NAN] * 3 + [None]],
                    columns=DATA_DIRECTORY['ENDSCALE'].specification.columns
                )
            )
        ),
        (
            '\n'.join((
                'TABDIMS',
                '2 4 2* 3',
            )),
            ValueError()
        ),
        (
            '\n'.join((
                'TABDIMS',
                '2 4 2* 3',
                'abc'
            )),
            ValueError()
        )
    ],
    DataTypes.PARAMETERS: [
        (
            '\n'.join((
                'RPTSOL',
                'RESTART=2 /'
                'abc'
            )),
            (
                'RPTSOL',
                {
                    'RESTART': '2',
                }
            )
        ),
        (
            '\n'.join((
                'RPTSOL',
                'RESTART=2',
                '/'
                'abc'
            )),
            (
                'RPTSOL',
                {
                    'RESTART': '2',
                }
            )
        ),
        (
            '\n'.join((
                'RPTSCHED',
                'FIP WELSPECS WELLS /',
                'abc'
            )),
            (
                'RPTSCHED',
                {
                    'FIP': None,
                    'WELSPECS': None,
                    'WELLS': None,
                }
            )
        ),
        (
            '\n'.join((
                'RPTSCHED',
                'FIP',
                'WELSPECS',
                'WELLS',
                '/',
                'abc'
            )),
            (
                'RPTSCHED',
                {
                    'FIP': None,
                    'WELSPECS': None,
                    'WELLS': None,
                }
            )
        ),
        (
            '\n'.join((
                'REPORTSCREEN',
                'WELL LOW',
                'ITERS MEDIUM',
                '/'
            )),
            (
                'REPORTSCREEN',
                {
                    'WELL': 'LOW',
                    'ITERS': 'MEDIUM',
                }
            )
        ),
        (
            '\n'.join((
                'REPORTSCREEN',
                'WELL LOW abc',
                'ITERS MEDIUM',
                '/'
            )),
            ValueError()
        ),
        (
            '\n'.join((
                'RPTSCHED',
                "'WELLS=2' 'SUMMARY=2' 'fip=3' 'RESTART=1' 'WELSPECS' 'CPU=2' /"
            )),
            (
                'RPTSCHED',
                {
                    'WELLS': '2',
                    'SUMMARY': '2',
                    'fip': '3',
                    'RESTART': '1',
                    'WELSPECS': None,
                    'CPU': '2'
                }
            )
        )

    ],
    DataTypes.ARRAY: [
        (
            '\n'.join((
                'ACTNUM',
                '3*0 2*1',
                '5*0',
                '/'
            )),
            (
                'ACTNUM',
                np.array([False]*3 + [True]*2 + [False]*5)
            )
        ),
        (
            '\n'.join((
                'PERMX',
                '3*0 2*1',
                '5*0.5 6',
                '/'
            )),
            (
                'PERMX',
                np.array([0]*3 + [1]*2 + [0.5]*5 + [6])
            )
        )
    ],
    DataTypes.OBJECT_LIST: [
        (
            '\n'.join((
                'WOPR',
                "'PROD1'",
                'PROD2',
                '/'
            )),
            (
                'WOPR',
                ['PROD1', 'PROD2'],
            )
        ),
        (
            '\n'.join((
                'WOPR',
                "'PROD1'",
                'PROD2/',
                ''
            )),
            (
                'WOPR',
                ['PROD1', 'PROD2'],
            )
        ),
        (
            '\n'.join((
                'WOPR',
                "'PROD1'",
                'PROD2',
                ''
            )),
            ValueError(),
        ),
        (
            '\n'.join((
                'WOPR',
                "'PROD1'",
                "",
                'PROD2',
                '/'
            )),
            ValueError(),
        ),
        (
            '\n'.join((
                'DATES',
                '01 JUL 2011/',
                '01 AUG 2012/',
                '/'
            )),
            (
                'DATES',
                [
                    pd.to_datetime('2011-07-01'),
                    pd.to_datetime('2012-08-01')
                ]
            )
        ),
        (
            '\n'.join((
                'DATES',
                '01 JUL 2011/',
                '01 AUG 2012',
                '/'
            )),
            ValueError()

        )
    ],
    DataTypes.RECORDS: [
        (
            '\n'.join((
                'TUNING',
                '1 365 0.1 0.15 3 0.3 0.1 1.25 0.75 /',
                '0.1 0.001 1E-7 0.0001',
                '10 0.01 1E-6 0.001 0.001 /',
                '12 1 25 1 8 8 4*1E6 /'
            )),
            (
                'TUNING',
                (
                    pd.DataFrame(
                        {key: value for key, value in zip(DATA_DIRECTORY['TUNING'].specification.specifications[0].columns,
                                                          [1.0, 365.0, 0.1, 0.15, 3.0, 0.3, 0.1, 1.25, 0.75,
                                                           np.NaN])}, index=[0]
                    ),
                    pd.DataFrame(
                        {key: value for key, value in zip(DATA_DIRECTORY['TUNING'].specification.specifications[1].columns,
                                                          [0.1, 0.001, 1e-7, 0.0001, 10.0, 0.01, 1e-6,
                                                           0.001, 0.001] + [np.NaN] * 3 + [INT_NAN])}, index=[0]
                    ),
                    pd.DataFrame(
                        {key: value for key, value in zip(DATA_DIRECTORY['TUNING'].specification.specifications[2].columns,
                                                          [12, 1, 25, 1, 8, 8] + [1e6]*4)}, index=[0]
                    )
                )
            )
        ),
        (
            '\n'.join((
                'TUNING',
                '1 365 0.1 0.15 3 0.3 0.1 1.25 0.75 /',
                '0.1 0.001 1E-7 0.0001',
                '10 0.01 1E-6 0.001 0.001 /',
            )),
            ValueError()
        ),
        (
            '\n'.join((
                'VFPPROD',
                "1 2200 'OIL' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /",
                '1 30 300 /',
                '10 20 /',
                '0 0.7 /',
                '1 100 500 /',
                '0 /',
                '1 1 1 1 1.75243E+002 1.75243E+002 1.75244E+002 /',
                '2 1 1 1 1.80749E+002 1.80749E+002 1.80750E+002 /',
                '1 2 1 1 1.91358E+002 1.91359E+002 1.91362E+002 /',
                '2 2 1 1 1.96743E+002 1.96744E+002 1.96747E+002 /' ,
                '1 1 2 1 1.71599E+002 1.71599E+002 1.71601E+002 /',
                '2 1 2 1 1.77093E+002 1.77093E+002 1.77095E+002 /',
                '1 2 2 1 1.88482E+002 1.88483E+002 1.88487E+002 /',
                '2 2 2 1 1.93865E+002 1.93866E+002 1.93869E+002 /',
                '1 1 3 1 1.45582E+002 1.45526E+002 1.45462E+002 /',
                '2 1 3 1 1.50977E+002 1.50978E+002 1.50979E+002 /',
                '1 2 3 1 1.71277E+002 1.71278E+002 1.71282E+002 /',
                '2 2 3 1 1.71277E+002 1.71278E+002 1.71282E+002 /',
            )),
            (
                'VFPPROD',
                (
                    pd.DataFrame([[
                        1, 2200.0, 'OIL', 'WCT', 'GOR', 'THP', ' ', 'METRIC', 'BHP',
                    ]], columns=['TABLE_NUM', 'BH_DATUM_DEPTH', 'FLO', 'WFR',
                                 'GFR', 'THP', 'ALQ', 'UNITS', 'QUANTITY']),
                    np.array([1, 30, 300]),
                    np.array([10, 20]),
                    np.array([0, 0.7]),
                    np.array([1, 100, 500]),
                    np.array([0]),
                    pd.DataFrame(
                        [[1, 1, 1, 1, 1.75243e2, 1.75243e2, 1.75244e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[2, 1, 1, 1, 1.80749e2, 1.80749e2, 1.80750e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 1, 2, 1, 1, 1.91358e2, 1.91359e2, 1.91362e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 2, 2, 1, 1, 1.96743e2, 1.96744e2, 1.96747e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 1, 1, 2, 1, 1.71599e2, 1.71599e2, 1.71601e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 2, 1, 2, 1, 1.77093e2, 1.77093e2, 1.77095e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 1, 2, 2, 1, 1.88482e2, 1.88483e2, 1.88487e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 2, 2, 2, 1, 1.93865e2, 1.93866e2, 1.93869e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 1, 1, 3, 1, 1.45582e2, 1.45526e2, 1.45462e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 2, 1, 3, 1, 1.50977e2, 1.50978e2, 1.50979e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 1, 2, 3, 1, 1.71277e2, 1.71278e2, 1.71282e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                    pd.DataFrame(
                        [[ 2, 2, 3, 1, 1.71277e2, 1.71278e2, 1.71282e2]],
                        columns=['NT', 'NW', 'NG', 'NA', 'BHP_THT1', 'BHP_THT2', 'BHP_THT3']
                    ),
                )
            )
        )
    ],
    DataTypes.STATEMENT_LIST: [
        (
            '\n'.join((
                'WCONPROD',
                '1043 OPEN LRAT 18.19 0 0 18.99 2* /',
                '1054 OPEN ORAT 16.38 1.765 0 18.14 1* 50 /',
                '/'
            )),
            (
                'WCONPROD',
                pd.DataFrame({key: (value, value2) for key, value, value2 in zip(
                    DATA_DIRECTORY['WCONPROD'].specification.columns,
                    ['1043', 'OPEN', 'LRAT', 18.19, 0.0, 0.0, 18.99, np.NaN, np.NaN, np.NaN, INT_NAN] +
                        [np.NaN] * 9,
                    ['1054', 'OPEN', 'ORAT', 16.38, 1.765, 0.0, 18.14, np.NaN, 50.0, np.NaN, INT_NAN] +
                        [np.NaN] * 9
                )}),
            )
        ),
        (
            '\n'.join((
                'WCONPROD',
                '1043 OPEN LRAT 18.19 0 0 18.99 2* /',
                '1054 OPEN ORAT 16.38 1.765 0 18.14 1* 50 /',
            )),
            ValueError()
        ),
        (
            '\n'.join((
                'WCONPROD',
                '1043 OPEN LRAT 18.19 0 0 18.99 2* /',
                '1054 OPEN ORAT 16.38 1.765 0 18.14 1* 50 /',
                '')),
            ValueError()
        ),
        (
            '\n'.join((
                'WELSPECS',
                "'3' 'GROUP 1' 22 20 1* 'OIL' 1* 1* 1* 1* 1* 1* 1* 1* 1* 1* 1* /",
                '/'
            )),
            (
                'WELSPECS',
                pd.DataFrame(
                    [['3', 'GROUP 1', 22, 20, np.NaN, 'OIL', np.NaN] + [None]*3 + [INT_NAN] + [None]*2],
                    columns=DATA_DIRECTORY['WELSPECS'].specification.columns
                )
            )
        ),
        (
            '\n'.join((
                'WECON',
                'P*    2*  0.9 2*  WELL/',
                '/'
            )),
            (
                'WECON',
                pd.DataFrame([[
                    'P*', np.NaN, np.NaN, 0.9, np.NaN, np.NaN,
                    'WELL', None, None, None,
                    np.NaN, None, np.NaN, np.NaN,
                ]], columns=DATA_DIRECTORY['WECON'].specification.columns)
            )
        )
    ],
    DataTypes.ARRAY_WITH_UNITS: [
        (
            '\n'.join((
                'RPTRSTT',
                'MONTH',
                '2 3 4',
                '/'
            )),
            (
                'RPTRSTT',
                ArrayWithUnits(
                    'MONTH',
                    np.array([2.0, 3.0, 4.0])
                )
            )
        )
    ],
    None: [
        (
            '\n'.join((
                'RPTRSTL',
                '/'
            )),
            (
                'RPTRSTL',
                None
            ),
        ),
        (
            '\n'.join((
                'RPTRSTL',
                'abc'
            )),
            ValueError()
        )
    ]
}


@pytest.mark.parametrize(
    "data_type, input, expected",
    itertools.chain(*([(key, val, exp) for (val,  exp) in data] for (key, data) in TEST_DATA.items()))
)
def test_load(data_type, input, expected):
    class IterPrev:
        def __init__(self, buf):
            self._buf = buf
            self._last = None
        def __iter__(self):
            return self
        def __next__(self):
            self._last = next(self._buf)
            return self._last
        def prev(self):
            if self._last is not(None):
                self._buf = itertools.chain((self._last,), self._buf)
                self._last = None
            else:
                raise ValueError('Can not get previous line.')
    buf = iter(input.splitlines())

    buf = IterPrev(buf)

    keyword = next(buf)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            LOADERS[data_type](DATA_DIRECTORY[keyword].specification, buf)
    else:
        res = LOADERS[data_type](DATA_DIRECTORY[keyword].specification, buf)
        if not isinstance(expected[1], tuple | list | ArrayWithUnits):
            expected_res = [expected[1]]
            res = [res]
        else:
            expected_res = expected[1]
        for r, e in zip(res, expected_res):
            if isinstance(e, np.ndarray):
                np.testing.assert_equal(r, e)
            elif isinstance(e, pd.DataFrame):
                pd.testing.assert_frame_equal(r, e)
            else:
                assert r == e

DECOMPRESS_TEST_DATA = [
    (
        ('3*1 2*0 1', bool),
        np.array([True]*3 + [False]*2 +[True])
    ),
    (
        ('3*1 2*0 1', int),
        np.array([1]*3 + [0]*2 +[1])
    ),
    (
        ('3*1 2*0 1', float),
        np.array([1.0]*3 + [0.0]*2 +[1.0])
    ),
]

@pytest.mark.parametrize(
    "inp, dtype, expected",
    [(inp, dtype, exp) for ((inp, dtype), exp) in DECOMPRESS_TEST_DATA]
)
def test_decompress(inp, dtype, expected):
    res = decompress_array(inp, dtype)
    np.testing.assert_equal(res, expected)

PARSE_VALS_TEST_DATA = [
    (
        ([None] * 10, 2, ['1', '2*5', '*', '3*']),
        ([None] * 2 + ['1'] + ['5']*2 + [None] * 5, 9)
    )
]

@pytest.mark.parametrize(
    'inp, expected',
    [value for value in PARSE_VALS_TEST_DATA]
)
def test_parse_vals(inp, expected):
    full, shift, vals = inp
    exp, exp_shift = expected
    res, res_shift = parse_vals(full, shift, vals)
    assert res == exp
    assert res_shift == exp_shift
