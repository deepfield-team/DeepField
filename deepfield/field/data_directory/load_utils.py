import logging
import uuid

import numpy as np

from .data_directory import DATA_DIRECTORY, DataTypes, DTYPES


DEFAULT_ENCODINGS = ['utf-8', 'cp1251']
def _load_string(keyword, buf):
    line = next(buf)
    split = line.split('/')
    val = split[0].strip(' \t\n\'\""')
    if len(split) == 0:
        line = next(buf)
        if not line.startswith('/'):
            raise ValueError(f'Data for keyword {keyword} was not properly terminated.')
    return val

def _load_vector(keyword, buf):
    line = next(buf)
    split = line.split('/')
    dtype = DTYPES[keyword] if keyword in DTYPES else float
    vector = np.fromstring(split[0], sep=' ', dtype=dtype)
    if len(split) == 0:
        line = next(buf)
        if not line.startswith('/'):
            raise ValueError(f'Data for keyword {keyword} was not properly terminated.')
    return vector




_LOADERS = {
    None: lambda keyword, buf: None,
    **{t: lambda keyword, buf: None for t in DataTypes},
    DataTypes.STRING: _load_string,
    DataTypes.VECTOR: _load_vector,
}
class StringIteratorIO:
    """String iterator for text files."""
    def __init__(self, path, encoding=None):
        self._path = path
        if (encoding is not None) and encoding.startswith('auto'):
            encoding = encoding.split(':')
            if len(encoding) > 1:
                n_bytes = int(encoding[1])
            else:
                n_bytes = 5000
            with open(self._path, 'rb') as file:
                raw = file.read(n_bytes)
                self._encoding = chardet.detect(raw)['encoding']
        else:
            self._encoding = encoding
        self._line_number = 0
        self._f = None
        self._buffer = ''
        self._last_line = None
        self._on_last = False
        self._proposed_encodings = DEFAULT_ENCODINGS.copy()

    @property
    def line_number(self):
        """Number of lines read."""
        return self._line_number

    def __iter__(self):
        return self

    def __next__(self):
        if self._on_last:
            self._on_last = False
            return self._last_line
        try:
            line = next(self._f).split('--')[0]
        except UnicodeDecodeError:
            return self._better_decoding()
        self._line_number += 1
        if line.strip():
            self._last_line = line
            return line
        return next(self)

    def _better_decoding(self):
        """Last chance to read line with default encodings."""
        try:
            enc = self._proposed_encodings.pop()
        except IndexError as err:
            raise UnicodeDecodeError('Failed to decode at line {}'.format(self._line_number + 1)) from err
        if enc == self._encoding:
            return self._better_decoding()
        self._f = open(self._path, 'r', encoding=enc) #pylint: disable=consider-using-with
        self._encoding = enc
        for _ in range(self._line_number):
            next(self._f)
        return next(self)

    def prev(self):
        """Set current position to previous line."""
        if self._on_last:
            raise ValueError("Maximum cache depth is reached.")
        self._on_last = True
        return self

    def __enter__(self):
        self._f = open(self._path, 'r', encoding=self._encoding) #pylint: disable=consider-using-with
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb
        self._f.close()

    def read(self, n=None):
        """Read n characters."""
        while not self._buffer:
            try:
                self._buffer = next(self)
            except StopIteration:
                break
        result = self._buffer[:n]
        self._buffer = self._buffer[len(result):]
        return result

    def skip_to(self, stop, *args):
        """Skip strings until stop token."""
        if isinstance(stop, str):
            stop = [stop]
        stop_pattern = '|'.join([x + '$' for x in stop])
        for line in self:
            if re.match(stop_pattern, line.strip(), *args):
                return

def load(path, logger=None, encoding=None):

    res = {
    }
    sections = DATA_DIRECTORY.keys()
    if logger is None:
        logger = logging.getLogger(str(uuid.uuid4()))
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    data_dir = path.parent
    filename = path.name

    logger.info(f'Start reading {filename}')
    cur_section = None
    with StringIteratorIO(path, encoding=encoding) as lines:
        for line in lines:
            firstword = line.split(maxsplit=1)[0].upper()
            if firstword in sections:
                cur_section = firstword
                if cur_section not in res:
                    res[cur_section] = []
                continue
            if cur_section is None:
                logger.warning(f'{firstword} is not a section name.')
                continue
            if firstword in DATA_DIRECTORY[cur_section]:
                logger.info(f'Reading keyword {firstword}.')
                data = _LOADERS[DATA_DIRECTORY[cur_section][firstword]](firstword, lines)
                res[cur_section].append((firstword, data))
            else:
                logger.warning('Keyword {firstword} in section {cur_section} is not supported.')

    return res






