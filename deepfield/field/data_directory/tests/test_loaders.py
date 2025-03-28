import itertools
import pytest
import numpy as np

from deepfield.field.data_directory.load_utils import LOADERS

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
        res = LOADERS[data_type](keyword, buf)
        if isinstance(expected[1], np.ndarray):
            np.testing.assert_equal(res, expected[1])
        else:
            assert (keyword, res) == expected






