import io
import itertools
import pandas as pd
import numpy as np
import pytest

from deepfield.field.data_directory.data_directory import INT_NAN, STATEMENT_LIST_INFO, DataTypes
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
                '/'
            ))
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
