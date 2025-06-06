import pandas as pd
import pytest
from deepfield.field.data_directory.data_directory import (SECTIONS, DataTypes,
    get_dynamic_keyword_specification, KeywordSpecification, TableSpecification)
TEST_DATA = (
    (
        (
            'ZMFVD',
            {
                'RUNSPEC': (
                    (
                        'COMPS',
                        pd.DataFrame([[5]], columns=['N'])
                    ),
                ),
            }
        ),
        KeywordSpecification('ZMFVD', DataTypes.TABLE_SET, TableSpecification(
            ['DEPTH'] + [f'C{i}' for i in range(1, 6)], [0], ['float'] * 6
        ), (SECTIONS.PROPS,)),
    ),
    (
        (
            'ZMFVD',
            {
                'RUNSPEC': (),
            }
        ),
        ValueError()
    ),
)

@pytest.mark.parametrize(
    'kw, data, expected',
    [(kw, data, expected) for ((kw, data), expected) in TEST_DATA]
)
def test_dynamic_specification(kw, data, expected):
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            res = get_dynamic_keyword_specification(kw, data)
    else:
        res = get_dynamic_keyword_specification(kw, data)
        assert res == expected
