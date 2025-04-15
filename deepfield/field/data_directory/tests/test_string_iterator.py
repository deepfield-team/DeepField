import pathlib

import pytest

from deepfield.field.data_directory.load_utils import StringIteratorIO

RESULT_LINES = (
    'line 1',
    'line 3',
    'inc1 line 1',
    'inc1 line 3',
    'line 7',
)

@pytest.fixture
def iterator():
    data_file_path = pathlib.Path(__file__).parent / 'data' / 'string_iterator_io_test_data' / 'test.data'
    return StringIteratorIO(data_file_path)

def test_iterator(iterator):
    with iterator:
        for line1, line2 in zip(iterator, RESULT_LINES):
            assert line1 == line2
