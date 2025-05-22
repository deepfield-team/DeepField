import io
import itertools
import pathlib
from string import Template
import pandas as pd
import numpy as np
import pytest

from deepfield.field.data_directory.data_directory import (DATA_DIRECTORY, INT_NAN, DataTypes,)
from deepfield.field.data_directory.dump_utils import DUMP_ROUTINES, dump
from deepfield.field.data_directory.load_utils import load

DUMP_ROUTINES_TEST_DATA = {
    DataTypes.TABLE_SET: [
        (
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
            ),
            '\n'.join((
                'SWOF',
                '0.42\t0.0\t0.737\t0.0',
                '0.48728\t0.000225\t0.610213\t0.0',
                '0.55456\t0.00438\t0.310527\t0.0',
                '0.62184\t0.023012\t0.072027\t0.0',
                '0.68912\t0.069122\t0.003178\t0.0',
                '0.7564\t0.151\t0.0\t0.0',
                '0.82368\t0.267672\t0.0\t0.0',
                '0.89096\t0.408671\t0.0\t0.0',
                '0.95824\t0.557237\t0.0\t0.0',
                '1.0\t0.645099\t0.0\t0.0',
                '/',
                '0.0\t0.0\t1.0\t0.0',
                '0.3\t0.002\t0.81\t0.0',
                '0.4\t0.018\t0.49\t0.0',
                '0.5\t0.05\t0.25\t0.0',
                '0.6\t0.098\t0.09\t0.0',
                '0.7\t0.162\t0.01\t0.0',
                '1.0\t0.2\t0.0\t0.0',
                '/'
            )),
        ),
        (
            (
                'EQUIL',
                (
                    pd.DataFrame(np.array([
                        [2300, 200, 2500, 0.1, 2300, 0.001, np.NaN, np.NaN, np.NaN],
                    ]), columns=DATA_DIRECTORY['EQUIL'].specification.columns),
                    pd.DataFrame(np.array([
                        [2310, 205, 2520, 0.05, 2310, 0.0, np.NaN, np.NaN, np.NaN],
                    ]), columns=DATA_DIRECTORY['EQUIL'].specification.columns),
                    pd.DataFrame(np.array([
                        [2305, 210, 2510, np.NaN, 2305, np.NaN, np.NaN, np.NaN, np.NaN],
                    ]), columns=DATA_DIRECTORY['EQUIL'].specification.columns)
                )
            ),
            '\n'.join((
                'EQUIL',
                '2300.0\t200.0\t2500.0\t0.1\t2300.0\t0.001',
                '/',
                '2310.0\t205.0\t2520.0\t0.05\t2310.0\t0.0',
                '/',
                '2305.0\t210.0\t2510.0\t*\t2305.0',
                '/'
            )),
        ),
        (
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
            ),
            '\n'.join((
                'PVTO',
                '1.0\t5.0\t1.031\t5.81\t/',
                '12.33\t52.0\t1.08\t5.03\t/',
                '21.65\t73.0\t1.1021\t4.23',
                '\t204.0\t1.092\t4.62',
                '\t321.0\t1.016\t6.02\t/',
                '/',
                '1.0\t5.0\t1.0002\t3.58\t/',
                '14.87\t58.0\t1.086\t2.93\t/',
                '27.7\t90.0\t1.113\t2.25',
                '\t234.0\t1.1\t2.88',
                '\t387.0\t1.121\t3.96\t/',
                '/',
                '1.0\t5.0\t1.0002\t3.58\t/',
                '18.67\t57.0\t1.073\t2.89\t/',
                '31.65\t88.0\t1.1083\t2.2',
                '\t248.0\t1.093\t2.57',
                '\t334.0\t1.073\t4.23\t/',
                '/'
            ))

        )
    ],
    DataTypes.SINGLE_STATEMENT: [
        (
            (
                'TABDIMS',
                pd.DataFrame(
                    np.array([
                        [2, 4] + 2*[INT_NAN] + [3] + 11*[INT_NAN]
                    ]),
                    columns=DATA_DIRECTORY['TABDIMS'].specification.columns
                )
            ),
            '\n'.join((
                'TABDIMS',
                '2\t4\t2*\t3',
                '/'
            ))
        )
    ],
    DataTypes.STATEMENT_LIST: [
        (
            (
                'WCONPROD',
                pd.DataFrame({key: (value, value2) for key, value, value2 in zip(
                    DATA_DIRECTORY['WCONPROD'].specification.columns,
                    ['1043', 'OPEN', 'LRAT', 18.19, 0.0, 0.0, 18.99, np.NaN, np.NaN, np.NaN, INT_NAN] +
                        [np.NaN] * 9,
                    ['1054', 'OPEN', 'ORAT', 16.38, 1.765, 0.0, 18.14, np.NaN, 50.0, np.NaN, INT_NAN] +
                        [np.NaN] * 9
                )}),
            ),
            '\n'.join((
                'WCONPROD',
                '1043\tOPEN\tLRAT\t18.19\t0.0\t0.0\t18.99/',
                '1054\tOPEN\tORAT\t16.38\t1.765\t0.0\t18.14\t*\t50.0/',
                '/'
            ))
        )
    ],
    DataTypes.RECORDS: [
        (
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
            ),
            '\n'.join((
                'TUNING',
                '1.0\t365.0\t0.1\t0.15\t3.0\t0.3\t0.1\t1.25\t0.75/',
                '0.1\t0.001\t1E-7\t0.0001\t10.0\t0.01\t1E-6\t0.001\t0.001/',
                '12\t1\t25\t1\t8\t8\t1000000.0\t1000000.0\t1000000.0\t1000000.0/\n'
            )),
        )
    ],
    DataTypes.ARRAY: [
        (
            (
                'ACTNUM',
                np.array([False]*3 + [True]*2 + [False]*5)
            ),
            (
                '\n'.join((
                    'INCLUDE',
                    '"$include_dir/ACTNUM.inc"',
                    '/'
                )),
                '\n'.join((
                    'ACTNUM',
                    '0 0 0 1 1 5*0',
                    '/'
                ))
            )
        ),
    ],
    DataTypes.OBJECT_LIST: [
        (
            (
                'WOPR',
                ['PROD1', 'PROD2']
            ),
            '\n'.join((
                'WOPR',
                'PROD1',
                'PROD2',
                '/'
            ))
        )
    ],
    DataTypes.PARAMETERS: [
        (
            (
                'RPTSCHED',
                {
                    'FIP': None,
                    'WELSPECS': None,
                    'WELLS': None
                },
            ),
            '\n'.join((
                'RPTSCHED',
                'FIP WELSPECS WELLS',
                '/'
            ))
        ),
        (
            (
                'RPTSOL',
                {
                    'RESTART': '2'
                },
            ),
            '\n'.join((
                'RPTSOL',
                'RESTART=2',
                '/'
            ))
        ),
        (
            (
                'REPORTSCREEN',
                {
                    'WELL': 'LOW',
                    'ITERS': 'MEDIUM'
                }
            ),
            '\n'.join(
                (
                    'REPORTSCREEN',
                    'WELL\tLOW',
                    'ITERS\tMEDIUM',
                    '/'
                )
            )
        )
    ],
    DataTypes.STRING: [
        (
            (
                'TITLE',
                'abc'
            ),
            '\n'.join((
                'TITLE',
                'abc',
                '/'
            ))
        )

    ],
    None: [
        (
            (
                'MULTOUT',
                None
            ),
            'MULTOUT'
        )
    ]
}

@pytest.mark.parametrize(
    'data_type, input, expected',
    itertools.chain(*([(key, val, exp) for (val,  exp) in data] for (key, data) in DUMP_ROUTINES_TEST_DATA.items()))
)
def test_dump_keyword(data_type, input, expected, tmp_path):
    with io.StringIO() as buf:
        DUMP_ROUTINES[data_type](DATA_DIRECTORY[input[0]], input[1], buf, tmp_path)
        result = buf.getvalue()
        if data_type == DataTypes.ARRAY:
            exp_buf = Template(expected[0]).safe_substitute(include_dir=tmp_path.name)
            assert result == exp_buf
            with open(tmp_path / f'{input[0]}.inc', 'r') as f:
                inc_res = f.read()
            assert inc_res == expected[1]
            return
        assert result == expected

def test_dump_load(tmp_path):
    egg_model_path = pathlib.Path('open_data/egg/Egg_Model_ECL.DATA')
    path = tmp_path/ 'egg_test'
    filename = 'Egg.data'
    data0 = load(egg_model_path)

    dump(data0, path=path, filename=filename)

    data1 = load(path / filename)
    for section in data0:
        for r, e in zip(data1[section], data0[section], strict=True):
            assert r[0] == e[0]
            if not isinstance(e[1], tuple | list):
                expected_res = [e[1]]
                res = [r[1]]
            else:
                expected_res = e[1]
                res = r[1]
            for r, e in zip(res, expected_res):
                if isinstance(e, np.ndarray):
                    np.testing.assert_equal(r, e)
                elif isinstance(e, pd.DataFrame):
                    pd.testing.assert_frame_equal(r, e)
                else:
                    assert  res == expected_res

