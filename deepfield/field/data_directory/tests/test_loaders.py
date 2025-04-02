import itertools
from typing import Sequence
import pytest
import numpy as np
import pandas as pd
from deepfield.field.data_directory.load_utils import LOADERS, TABLE_INFO, decompress_array

from deepfield.field.data_directory.data_directory import DataTypes

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
        )
    ],
    DataTypes.VECTOR: [
        (
            '\n'.join(('DIMENS', '1 1 1/', '')),
            ('DIMENS', np.array([1, 1, 1], dtype=int))
        ),
        (
            '\n'.join(('DIMENS', '1 1 1', '/', '')),
            ('DIMENS', np.array([1, 1, 1], dtype=int))
        ),
        (
            '\n'.join(('DIMENS', '1 1 1', '', '')),
            ValueError()
        ),
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
        )
    ]
}


@pytest.mark.parametrize(
    "data_type, input, expected",
    itertools.chain(*([(key, val, exp) for (val,  exp) in data] for (key, data) in TEST_DATA.items()))
)
def test_load(data_type, input, expected):
    buf = iter(input.splitlines())
    keyword = next(buf)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            LOADERS[data_type](keyword, buf)
    else:
        print(keyword)
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

