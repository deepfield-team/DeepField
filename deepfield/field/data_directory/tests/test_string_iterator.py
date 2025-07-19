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

def test_current_file(iterator):
    with iterator:
        for line in iterator:
            if line.startswith('inc1'):
                assert iterator.current_file.absolute() == (
                        pathlib.Path(__file__).parent /
                            'data' /
                            'string_iterator_io_test_data' /
                            'inc1.inc').absolute()
            else:
                assert iterator.current_file.absolute() == (
                        pathlib.Path(__file__).parent /
                            'data' /
                            'string_iterator_io_test_data' /
                            'test.data').absolute()

def test_line_number(iterator):
    with iterator:
        assertions = [False, False]
        for line in iterator:
            if line == 'line 3':
                assert iterator.line_number == 3
                assertions[0] = True
            if line == 'inc1 line 1':
                assert iterator.line_number == 1
                assertions[1] = True

    for a in assertions:
        assert a
