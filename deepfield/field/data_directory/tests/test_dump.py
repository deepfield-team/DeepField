import io
import itertools
import pandas as pd
import numpy as np
import pytest

from deepfield.field.data_directory.data_directory import INT_NAN, STATEMENT_LIST_INFO, DataTypes, RECORDS_INFO
from deepfield.field.data_directory.dump_utils import DUMP_ROUTINES

DUMP_ROUTINES_TEST_DATA = {
    DataTypes.SINGLE_STATEMENT: [
        (
            (
                'TABDIMS',
                pd.DataFrame(
                    np.array([
                        [2, 4] + 2*[INT_NAN] + [3] + 11*[INT_NAN]
                    ]),
                    columns=STATEMENT_LIST_INFO['TABDIMS']['columns']
                )
            ),
            '\n'.join((
                'TABDIMS',
                '2\t4\t2*\t3',
                '/\n'
            ))
        )
    ],
    DataTypes.STATEMENT_LIST: [
        (
            (
                'WCONPROD',
                pd.DataFrame({key: (value, value2) for key, value, value2 in zip(
                    STATEMENT_LIST_INFO['WCONPROD']['columns'],
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
                '/\n'
            ))
        )
    ],
    DataTypes.RECORDS: [
        (
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
            ),
            '\n'.join((
                'TUNING',
                '1.0\t365.0\t0.1\t0.15\t3.0\t0.3\t0.1\t1.25\t0.75/',
                '0.1\t0.001\t1E-7\t0.0001\t10.0\t0.01\t1E-6\t0.001\t0.001/',
                '12.0\t1.0\t25.0\t1.0\t8.0\t8.0\t1000000.0\t1000000.0\t1000000.0\t1000000.0/\n'
            )),
        )
    ]
}

@pytest.mark.parametrize(
    'data_type, input, expected',
    itertools.chain(*([(key, val, exp) for (val,  exp) in data] for (key, data) in DUMP_ROUTINES_TEST_DATA.items()))
)
def test_dump_keyword(data_type, input, expected):
    with io.StringIO() as buf:
        DUMP_ROUTINES[data_type](input[0], input[1], buf)
        result = buf.getvalue()
        assert result == expected
