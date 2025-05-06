import itertools
from typing import Sequence
import pytest
import numpy as np
import pandas as pd
from deepfield.field.data_directory.load_utils import LOADERS, TABLE_INFO, decompress_array, parse_vals

from deepfield.field.data_directory.data_directory import RECORDS_INFO, STATEMENT_LIST_INFO, DataTypes
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
            ValueError()
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
            ValueError()
        ),
        (
            '\n'.join(('INCLUDE', "'abc/abc'a", '/')),
            ValueError()
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
                    pd.DataFrame(np.array([
                        [2300, 200, 2500, 0.1, 2300, 0.001, np.NaN, np.NaN],
                    ]), columns=TABLE_INFO['EQUIL']['attrs']),
                    pd.DataFrame(np.array([
                        [2310, 205, 2520, 0.05, 2310, 0.0, np.NaN, np.NaN],
                    ]), columns=TABLE_INFO['EQUIL']['attrs']),
                    pd.DataFrame(np.array([
                        [2305, 210, 2510, np.NaN, 2305, np.NaN, np.NaN, np.NaN],
                    ]), columns=TABLE_INFO['EQUIL']['attrs'])
                )
            )
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
                    ), columns=TABLE_INFO['SWOF']['attrs']).set_index(
                        TABLE_INFO['SWOF']['attrs'][TABLE_INFO['SWOF']['domain'][0]]
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
                    ), columns=TABLE_INFO['SWOF']['attrs']).set_index(
                        TABLE_INFO['SWOF']['attrs'][TABLE_INFO['SWOF']['domain'][0]]
                    )
                )
            )
        ),
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
                    columns=STATEMENT_LIST_INFO['TABDIMS']['columns']
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
                    columns=STATEMENT_LIST_INFO['TABDIMS']['columns']
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
                        {key: value for key, value in zip(RECORDS_INFO['TUNING'][0]['columns'],
                                                          [1.0, 365.0, 0.1, 0.15, 3.0, 0.3, 0.1, 1.25, 0.75,
                                                           np.NaN])}, index=[0]
                    ),
                    pd.DataFrame(
                        {key: value for key, value in zip(RECORDS_INFO['TUNING'][1]['columns'],
                                                          [0.1, 0.001, 1e-7, 0.0001, 10.0, 0.01, 1e-6,
                                                           0.001, 0.001] + [np.NaN] * 3 + [INT_NAN])}, index=[0]
                    ),
                    pd.DataFrame(
                        {key: value for key, value in zip(RECORDS_INFO['TUNING'][2]['columns'],
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
                    STATEMENT_LIST_INFO['WCONPROD']['columns'],
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
            LOADERS[data_type](keyword, buf)
    else:
        res = LOADERS[data_type](keyword, buf)
        if not isinstance(expected[1], tuple | list):
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
                assert (keyword, res) == (expected[0], expected_res)

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
