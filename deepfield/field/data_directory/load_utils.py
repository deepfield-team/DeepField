from contextlib import ExitStack
import copy
import logging
import pathlib
import shlex
import uuid
import re

import numpy as np
import pandas as pd

from .data_directory import DATA_DIRECTORY, INT_NAN, STATEMENT_LIST_INFO, TABLE_INFO, DataTypes, DTYPES


DEFAULT_ENCODINGS = ['utf-8', 'cp1251']
def _load_string(keyword, buf):
    line = next(buf)
    split = line.split('/')
    val = split[0].strip(' \t\n\'\""')
    if len(split) == 1:
        line = next(buf)
        if not line.startswith('/'):
            raise ValueError(f'Data for keyword {keyword} was not properly terminated.')
    return val

def _load_vector(keyword, buf):
    line = next(buf)
    split = line.split('/')
    dtype = DTYPES[keyword] if keyword in DTYPES else float
    vector = np.fromstring(split[0], sep=' ', dtype=dtype)
    if len(split) == 1:
        line = next(buf)
        if not line.startswith('/'):
            raise ValueError(f'Data for keyword {keyword} was not properly terminated.')
    return vector

def _load_table(keyword, buf):
    table_info = TABLE_INFO[keyword]
    n_attrs = len(table_info['attrs'])
    dtype = table_info['dtype'] if 'dtype' in table_info else float
    data = _read_numerical_table_data(buf, 1, dtype)
    tables = []
    for region_table_data in data:
        if region_table_data.size < n_attrs:
            tmp = np.empty(n_attrs - region_table_data.size, dtype=dtype)
            if dtype == float:
                tmp[:] = np.nan
            elif dtype == int:
                tmp[:] = INT_NAN
            else:
                tmp[:] = None
            region_table_data = np.concatenate((region_table_data, tmp))
        data_tmp = region_table_data.reshape(-1, n_attrs)
        table = pd.DataFrame(data_tmp, columns=table_info['attrs'])
        if 'domain' in table_info and table_info['domain'] is not None:
            if len(table_info['domain'])==1:
                domain = table_info['attrs'][table_info['domain'][0]]
            else:
                raise ValueError('Single attribute should be specified for table index.')
            table = table.set_index(domain)
        tables.append(table)
    return tables


def _read_numerical_table_data(buffer, depth, dtype):
    """Read numerical data for table.

    Parameters
    ----------
    buffer : StringIteratorIO
        String buffer to read.
    depth : _type_
        Depth of the table nesting (2 for multiindex table, 1 for normal table)
    dtype : _type_
        Data dtype.

    Returns
    -------
    List[np.ndarray] or List[List[np.ndarray]]
        List of numpy arrays (1 array for each region), if `depth==1`.
        List of lists of numpy array (1 array for each subtable, list of arrays
        for each region), if depth==2

    Raises
    ------
    ValueError
        If table block is not properly closed
    """
    data = []
    for _ in range(depth):
        data = list(data)
    ind = [0] * depth
    group_end = True
    expr = re.compile(r'(\d*)\*')

    def _repl(match):
        num = match.groups()[0]
        num = int(num) if num else 1
        return ' '.join(['nan']*num)

    for line in buffer:
        line = line.strip()
        split = line.split("/")
        line = split[0]
        if len(line) > 0:
            cur_item = data
            line = expr.sub(_repl, line)
            for i in reversed(ind):
                if len(cur_item) == i:
                    cur_item.append([])
                cur_item = cur_item[i]
            if not (line[0].isdigit() or line[0]=='.' or line[:3]=='nan'): # line can start from `.`, e.g `.123`
                buffer.prev()
                break
            line = line.replace('nan', str(INT_NAN))
            numbers = np.fromstring(line, dtype=dtype, sep=' ')
            cur_item.append(numbers)
            group_end = False
        if len(split) > 1:
            if group_end:
                try:
                    ind[1] += 1
                except IndexError:
                    data.append([])
                    buffer.prev()
                    break
                ind[0] = 0
            else:
                ind[0] += 1
            group_end = True
    if data[-1] and (len(data[-1][0])):
        if ind[-1] == len(data)-1:
            raise ValueError('Table block was not properly closed.')
    else:
        del data[-1]
        ind[-1] -= 1

    if depth == 1:
        tmp_iter = [data]
    else:
        tmp_iter = data
    for d in (tmp_iter):
        for i, vals in enumerate(d):
            d[i] = np.hstack(vals)
    return data

def _load_array(keyword, buf):
    kwargs = {}
    if keyword in DTYPES:
        kwargs['dtype'] = DTYPES[keyword]
    data = read_array(buf, **kwargs)
    return data

def read_array(buffer, dtype=None, compressed=True, **kwargs):
    """Read array data from a string buffer before first occurrence of '/' symbol.

    Parameters
    ----------
    buffer : buffer
        String buffer to read.
    dtype : dtype or None
        Defines dtype of an output array. If not specified, float array is returned.
    compressed : bool
        If True, A*B will be interpreted as B repeated A times.

    Returns
    -------
    arr : ndarray
        Parsed array.
    """
    _ = kwargs
    arr = []
    last_line = False
    if dtype is None:
        dtype = float
    for line in buffer:
        if '/' in line:
            last_line = True
            line = line.split('/')[0]
        if compressed:
            x = decompress_array(line, dtype=dtype)
        else:
            x = np.fromstring(line.strip(), dtype=dtype, sep=' ')
        if x.size:
            arr.append(x)
        if last_line:
            break
    return np.hstack(arr)

def decompress_array(s, dtype=None):
    """Extracts compressed numerical array from ASCII string.
    Interprets A*B as B repeated A times."""
    if dtype is None:
        dtype = float
    nums = []
    for x in s.split():
        try:
            val = [dtype(float(x))]
        except ValueError:
            k, val = x.split('*')
            val = [dtype(float(val))] * int(k)
        nums.extend(val)
    return np.array(nums)

def _load_parameters(keyword, buf):
    res = {}
    for line in buf:
        split = line.split('/')
        words = split[0].split()
        for word in words:
            if '=' in word:
                key, val = word.split('=')
                res[key] = val
            else:
                res[word] = None

        if len(split) > 1:
            break
    return res

def parse_vals(columns, shift, full, vals):
    """Parse values (unpack asterisk terms)."""
    full = copy.deepcopy(full)
    for i, v in enumerate(vals):
        if i + shift >= len(columns):
            break
        if '*' in v:
            v = v.strip('\'\"')
            if v == '*':
                continue
            try:
                shift += int(v.strip('*')) - 1
            except ValueError:
                full[i+shift] = v
        else:
            full[i+shift] = v
    return full

def _load_statement_list(keyword, buf):
    """Parse Eclipse keyword data to dataframe.

    Parameters
    ----------
    buffer : StringIteratorIO
        Buffer to read data from.
    columns : list
        Keyword columns.
    column_types : dict
        Types of values in corrsponding columns.
    defaults : dict, optional
        Dictionary with default values, by default None.
    date : datetime, optional
        Date to be included in the output DataFrame.

    Returns
    -------
    pd.Dataframe
        Loaded keyword dataframe.
    """
    columns = STATEMENT_LIST_INFO[keyword]['columns']
    column_types = STATEMENT_LIST_INFO[keyword]['dtypes']
    df = pd.DataFrame(columns=columns)
    for line in buf:
        if '/' not in line:
            break
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = line.split()[:len(columns)]
        full = [None] * len(columns)
        full = parse_vals(columns, 0, full, vals)
        df = pd.concat([df, pd.DataFrame(dict(zip(columns, full)), index=[0])], ignore_index=True)

    if 'text' in column_types:
        text_columns = [col for col, dt in zip(columns, column_types) if dt=='text']
        df[text_columns] = df[text_columns].map(
            lambda x: x.strip('\'\"') if x is not None else x)
    if 'float' in column_types:
        float_columns = [col for col, dt in zip(columns, column_types) if dt=='float']
        df[float_columns] = df[float_columns].astype(float )
    if 'int' in column_types:
        int_columns = [col for col, dt in zip(columns, column_types) if dt=='int']
        df[int_columns] = df[int_columns].fillna(INT_NAN).astype(int)
    return df

LOADERS = {
    None: lambda keyword, buf: None,
    **{t: lambda keyword, buf: None for t in DataTypes},
    DataTypes.STRING: _load_string,
    DataTypes.VECTOR: _load_vector,
    DataTypes.TABLE_SET: _load_table,
    DataTypes.ARRAY: _load_array,
    DataTypes.PARAMETERS: _load_parameters,
    DataTypes.STATEMENT_LIST: _load_statement_list
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
        self._include = None
        self._on_last = False
        self._proposed_encodings = DEFAULT_ENCODINGS.copy()

    @property
    def line_number(self):
        """Number of lines read."""
        return self._line_number

    def __iter__(self):
        return self

    def __next__(self):
        if self._include is not None:
            try:
                return next(self._include)
            except StopIteration:
                self._include = None

        if self._on_last:
            self._on_last = False
            return self._last_line
        try:
            line = next(self._f).split('--')[0].strip()
        except UnicodeDecodeError:
            return self._better_decoding()
        self._line_number += 1
        if line:
            if line == 'INCLUDE':
                path = LOADERS[DataTypes.STRING]('INCLUDE', self)
                self.include_file(path)
                return next(self)
            self._last_line = line
            print(line)
            return line
        return next(self)

    def include_file(self, path):
        path = self._path.parent.joinpath(path)
        with self._stack as stack:
            self._include = stack.enter_context(StringIteratorIO(path, self._encoding))
            self._stack = stack.pop_all()

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
        with ExitStack() as stack:
            self._f = stack.enter_context(open(self._path, 'r', encoding=self._encoding)) #pylint: disable=consider-u
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb
        self._stack.close()

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
                data = LOADERS[DATA_DIRECTORY[cur_section][firstword]](firstword, lines)
                res[cur_section].append((firstword, data))
            else:
                logger.warning('Keyword {firstword} in section {cur_section} is not supported.')

    return res
