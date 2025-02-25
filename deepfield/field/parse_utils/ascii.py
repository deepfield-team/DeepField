"""Parser utils."""
import copy
import os
import re
import shlex
from io import StringIO
from itertools import zip_longest
from pathlib import Path

import chardet
import numpy as np
import pandas as pd

INT_NAN = -99999999

_COLUMN_LENGTH = 13


DEFAULT_ENCODINGS = ['utf-8', 'cp1251']


IGNORE_SECTIONS = [('ARITHMETIC',),
                   ('RPTISOL', 'RPTPROPS', 'RPTREGS', 'RPTRST',
                   'RPTRUNSP', 'RPTSCHED', 'RPTSMRY', 'RPTSOL',
                   'OUTSOL'),
                   ('FRACTURE_ARITHMETIC',)]
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

def preprocess_path(path):
    """Parse a string with path to Path instance."""
    parts = path.split('.')
    try:
        path_str = '.'.join(parts[:-1] + [parts[-1].rstrip(' /').split()[0]])
    except IndexError:
        return Path(path)
    path_str = path_str.strip(' \t\n\'"').replace('\\', '/')
    return Path(path_str)

def _insensitive_match(where, what):
    """Find unique 'what' in directory 'where' ignoring 'what' case."""
    found = [p for p in os.listdir(where) if p.lower() == what.lower()]
    if len(found) > 1:
        raise FileNotFoundError("Multiple paths found for {} in {}".format(what, where))
    if len(found) == 0:
        raise FileNotFoundError("Path {} does not exists in {}".format(what, where))
    return found[0]

def case_insensitive_path(path):
    """Resolve system path given path in arbitrary case."""
    parts = path.parts
    result = Path(parts[0])
    for p in parts[1:]:
        result = result / Path(_insensitive_match(str(result), p))
    return result

def _get_path(line, data_dir, logger, raise_errors):
    """Case insensitive file path parser."""
    path = preprocess_path(line)
    look_path = (data_dir / path).resolve()
    try:
        actual_path = case_insensitive_path(look_path)
    except FileNotFoundError as err:
        if raise_errors:
            raise FileNotFoundError(err) from None
        logger.warning("Ignore missing file {}.".format(str(look_path)))
        return None
    return actual_path

def tnav_ascii_parser(path, loaders_map, logger, data_dir=None, encoding=None, raise_errors=False):  # pylint: disable=too-many-branches, too-many-statements
    """Read tNav ASCII files and call loaders."""
    data_dir = path.parent if data_dir is None else data_dir
    filename = path.name
    for sections_to_ignore, loader in zip(IGNORE_SECTIONS,
                                          (_dummy_loader, _dummy_loader2, _dummy_loader3)):
        for keyword in sections_to_ignore:
            loaders_map[keyword] = loader
    logger.info("Start reading {}".format(filename))
    with StringIteratorIO(path, encoding=encoding) as lines:
        for line in lines:
            firstword = line.split(maxsplit=1)[0].upper()
            if firstword in ['EFOR', 'EFORM', 'HFOR', 'HFORM']:
                column_names = line.split()[1:]
            elif firstword == 'ETAB':
                if 'ETAB' in loaders_map:
                    logger.info("[{}:{}] Loading ETAB".format(filename, lines.line_number))
                    loaders_map['ETAB'](lines, column_names=column_names)
                else:
                    lines.skip_to(['/', 'ENDE'])
            elif firstword == 'TTAB':
                if 'TTAB' in loaders_map:
                    logger.info("[{}:{}] Loading TTAB".format(filename, lines.line_number))
                    loaders_map['TTAB'](lines)
                else:
                    lines.skip_to('ENDT')
            elif (firstword in ['EFIL', 'EFILE', 'TFIL']) and (firstword in loaders_map):
                line = next(lines)
                include = _get_path(line, data_dir, logger, raise_errors)
                if include is None:
                    continue
                with StringIteratorIO(include, encoding=encoding) as inc_lines:
                    logger.info("[{0}:{1}] Loading {2} from {3}"\
                                    .format(filename, lines.line_number, firstword, include))
                    if firstword == 'TFIL':
                        loaders_map[firstword](inc_lines)
                    else:
                        loaders_map[firstword](inc_lines, column_names=column_names)
            elif firstword == 'HTAB' and firstword in loaders_map:
                logger.info("[{}:{}] Loading HTAB".format(filename, lines.line_number))
                loaders_map['HTAB'](lines, column_names=column_names)
            elif (firstword in ['HFIL', 'HFILE']) and (firstword in loaders_map):
                line = next(lines)
                include = _get_path(line, data_dir, logger, raise_errors)
                if include is None:
                    continue
                with StringIteratorIO(include, encoding=encoding) as inc_lines:
                    logger.info("[{0}:{1}] Loading {2} from {3}"\
                                    .format(filename, lines.line_number, firstword, include))
                    loaders_map[firstword](inc_lines, column_names=column_names)
            elif firstword in ['INCLUDE', 'USERFILE']:
                line = next(lines)
                include = _get_path(line, data_dir, logger, raise_errors)
                if include is None:
                    continue
                logger.info("[{0}:{1}] Include {2}".format(filename, lines.line_number, include))
                tnav_ascii_parser(include, loaders_map, logger, data_dir=data_dir,
                                  encoding=encoding, raise_errors=raise_errors)
            elif (firstword == 'WELLTRACK') and (firstword in loaders_map):
                lines.prev() #pylint: disable=not-callable
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif (firstword in ['GROU', 'GROUP']) and (firstword in loaders_map):
                lines.prev() #pylint: disable=not-callable
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif (firstword in ['AQCO', 'AQCT']) and (firstword in loaders_map):
                lines.prev() #pylint: disable=not-callable
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif (firstword in ['AQUANCON', 'AQUCT']) and (firstword in loaders_map):
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif firstword in loaders_map:
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
    logger.info("Finish reading {}".format(filename))

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
            val = [dtype(val)] * int(k)
        nums.extend(val)
    return np.array(nums)

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
        for i, dd in enumerate(d):
            d[i] = np.hstack(dd)
    return data

def read_table(buffer, table_info, dtype=None, units='METRIC'):
    """Read numerical table data from a string buffer before first occurrence of non-digit line.

    Parameters
    ----------
    buffer: buffer
        String buffer to read.
    table_info: dict
        Dict with table's meta information:
            table_info['attrs'] - list of column names
            table_info['domain'] - list of domain columns indices
    dtype: dtype or None
        Defines dtype of an output array. If not specified, float array is returned.

    Returns
    -------
    table : pandas DataFrame
        Parsed table.
    """
    n_attrs = len(table_info['attrs'])

    if dtype is None:
        dtype = float

    depth = 2 if table_info['domain'] and len(table_info['domain'])==2 else 1
    data = _read_numerical_table_data(buffer, dtype=dtype, depth=depth)
    tables = []
    for region_table_data in data:
        if depth == 2:
            table_parts = []
            for d in region_table_data:
                data_tmp = d[1:].reshape(-1, n_attrs-1)
                data_tmp = np.hstack(
                    (np.ones((data_tmp.shape[0], 1), dtype=data_tmp.dtype) * d[0],
                     data_tmp))
                table_parts.append(data_tmp)
            table = pd.DataFrame(np.vstack(table_parts), columns=table_info['attrs'])
        else:
            if region_table_data.size < n_attrs:
                tmp = np.empty(n_attrs - region_table_data.size)
                tmp[:] = np.nan
                region_table_data = np.concatenate((region_table_data, tmp))
            data_tmp = region_table_data.reshape(-1, n_attrs)
            table = pd.DataFrame(data_tmp, columns=table_info['attrs'])


        if table_info['defaults']:
            for col, default in zip(table_info['attrs'], table_info['defaults']):
                if default:
                    if hasattr(default, '__iter__'):
                        val = default[0] if units=='METRIC' else default[1]
                    else:
                        val = default
                    table[col] = table[col].fillna(val)

        if table_info['domain'] is not None:
            domain_attrs = np.array(table_info['attrs'])[table_info['domain']]
        else:
            domain_attrs = np.array([])
        if domain_attrs.shape[0] == 1:
            table = table.set_index(domain_attrs[0])
        elif domain_attrs.shape[0] > 1:
            multi_index = pd.MultiIndex.from_frame(table[domain_attrs])
            table = table.drop(domain_attrs, axis=1)
            table = table.set_index(multi_index)
        tables.append(table)
    return tables[0] # return table for the first region

def read_rsm(filename, logger):
    """Parse *.RSM files to dict."""
    result = {}
    blocks = _rsm_blocks(filename)
    for block in blocks:
        block_res = _parse_block(block, logger)
        result = _update_result(result, block_res, logger)
    return result

def _rsm_blocks(filename):
    """TBD."""
    block = None
    block_start_re = re.compile(r'\d+\n')
    with open(filename) as f:
        line = f.readline()
        while line:
            if block_start_re.fullmatch(line):
                if block is not None:
                    yield ''.join(block)
                block = []
            if block is not None:
                block.append(line)
            line = f.readline()
        if block is not None:
            yield ''.join(block)

def _split_block(block):
    """TBD."""
    lines = block.split('\n')
    border_re = re.compile(r'\s\-+')
    border_count = 0

    i = None

    for i, line in enumerate(lines):
        if border_re.fullmatch(line):
            border_count += 1
        if border_count == 3:
            break

    if border_count == 3:
        return '\n'.join(lines[:i+1]), '\n'.join(lines[i+1:])
    raise ValueError('Block can not be splitted')

def _parse_header(header):
    """TBD."""
    header_data = header.split('\n')[4:-1]
    names = _split_string(header_data[0])
    units = _split_string(header_data[1])
    has_multiplyers = len(header_data[2].strip()) > 0 and header_data[2].strip()[0] == '*'
    multiplyers = (_split_string(header_data[2]) if has_multiplyers
                   else _split_string(''.join('\t' * len(header_data[2]))))
    multiplyers = [_parse_rsm_multiplyer(mult) for mult in multiplyers]
    obj_string_number = 3 if has_multiplyers else 2
    objects = _split_string(header_data[obj_string_number])
    numbers = (_split_string(header_data[obj_string_number+1]) if
               len(header_data) > (obj_string_number + 1) else [''] * len(objects))

    return names, units, multiplyers, objects, numbers

def _split_string(string, n_sym=_COLUMN_LENGTH):
    """TBD."""
    return [''.join(s).strip() for s  in grouper(string, n_sym)]

def _parse_rsm_multiplyer(multiplyer):
    """Parse rsm multiplyer value"""
    if multiplyer == '':
        return 1
    match = re.fullmatch(r'\*(\d+)\*\*(\d+)', multiplyer)
    if match is not None:
        return int(match.groups()[0]) ** int(match.groups()[1])
    raise ValueError('Wrong `multiplyer` format')

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def _parse_data(data):
    """TBD."""
    data = pd.read_csv(StringIO(data), header=None, sep=r'\s+')
    return  data

def _parse_block(block, logger):
    """TBD."""
    header, data = _split_block(block)
    names, units, multiplyers, objects, numbers = _parse_header(header)
    num_data = _parse_data(data)

    res = {}

    for i, (
            obj, name, unit, multiplyer, number
        ) in enumerate(zip(objects, names, units, multiplyers, numbers)):
        if obj == '':
            obj = '_global'

        if obj not in res:
            res[obj] = {}
        current_obj = res[obj]
        if number != '':
            if '_children' not in res[obj]:
                res[obj]['_children'] = {}
            if number not in res[obj]['_children']:
                res[obj]['_children'][number] = {}
            current_obj = res[obj]['_children'][number]

        if name not in current_obj:
            values = (num_data[i].apply(int, base=16).values.reshape(-1) if name.startswith('#')
                      else num_data[i].values.reshape(-1))
            current_obj[name] = {
                'units': unit,
                'multiplyer': multiplyer,
                'data': values
            }

        else:
            logger.warn(('Object {} already contains field {}. \
                          New data is ignored.').format(obj, name))

    return res

def _update_result(result, new_data, logger):
    """TBD."""
    for obj, obj_data in new_data.items():
        if obj not in result:
            result[obj] = obj_data
        else:
            if isinstance(result[obj], dict):
                result[obj] = _update_result(result[obj], obj_data, logger)
            else:
                is_equal = ((obj_data == result[obj]).all()
                            if isinstance(obj_data, np.ndarray)
                            else obj_data == result[obj])
                if not is_equal:
                    logger.warn(('New value of {} {} is not equal to old value {}. \
                                 Data was rewrited.').format(obj, obj_data, result[obj]))
                    result[obj] = obj_data
    return result

def parse_perf_line(line, column_names, defaults):
    """Get wellname and perforation from a single event file line.
    Expected format starts with: WELL 'DD.MM.YYYY' PERF/PERFORATION
    """
    vals = line.split()
    well = vals[0].strip("'\"")
    date = pd.to_datetime(vals[1], format='%d.%m.%Y', errors='coerce')
    vals = [v.upper() for v in vals]
    if 'BRANCH' in vals[3:]:
        well = ":".join([well, vals[vals.index('BRANCH') + 1]])
    data = {'WELL': well, 'DATE': date}
    for i, v in enumerate(vals[3:len(column_names) + 1]):
        if '*' in v:
            k = int(v.split('*', 1)[0])
            for j in range(k):
                name = column_names[2 + i + j]
                data[name] = [defaults[name]]
        else:
            name = column_names[2 + i]
            data[name] = [float(v)]
    data['CLOSE'] = 'CLOSE' in vals[3:]
    return pd.DataFrame(data)

def parse_control_line(line, mode_control, value_control):
    """Get wellname and control from a single event file line.
    Expected format starts with: WELL 'DD.MM.YYYY'
    """
    vals = line.split()
    data = {'WELL' : [vals[0].strip("'\"")],
            'DATE': [pd.to_datetime(vals[1], format='%d.%m.%Y', errors='coerce')]}
    vals = [v.upper() for v in vals]
    mode = [k for k in vals[2:] if k in mode_control]
    if mode:
        if len(mode) > 1:
            raise ValueError("Multiple mode controls.")
        data['MODE'] = [mode[0]]
    for i, k in enumerate(vals[2:]):
        if k in value_control:
            try:
                data[k] = [float(vals[i + 3])]
            except ValueError:
                data[k] = None
    return pd.DataFrame(data)

def parse_history_line(line, column_names):
    """Get data from a single history file line.
    Expected format starts with: WELL 'DD.MM.YYYY'
    """
    vals = line.split()
    well = vals[0].strip("'\"")
    date = pd.to_datetime(vals[1], format='%d.%m.%Y', errors='coerce')
    vals = [v.upper() for v in vals]
    data = {'WELL': well, 'DATE': date}
    for i, name in enumerate(column_names[2:]):
        data[name] = [float(vals[2 + i])]
    return pd.DataFrame(data)

def read_dates_from_buffer(buffer, attr, logger):
    """Read keywords representing output dates (ARRAY, DATES).

    Parameters
    ----------
    buffer: buffer
        String buffer to read from.
    attr: str
        Keyword to read.
    logger: Logger
        Log info handler.

    Returns
    -------
    output_dates: list
    """
    _ = attr
    buffer = buffer.prev()
    args = next(buffer).strip('\n').split()[1:]

    if args[0] != 'DATE':
        logger.warning('ARRAy of type {} is not supported and is ignored.'.format(args[0]))
        return None

    dates = []
    for line in buffer:
        if '/' in line:
            break
        dates.append(line)
    return pd.to_datetime(dates)

def dates_to_str(dates):
    """Transforms list of dates into a string representation with the ARRAY keyword heading.

    Parameters
    ----------
    dates: list

    Returns
    -------
    dates: str
    """
    heading = 'ARRAY DATE'
    footing = '/'
    res = [heading] + [d.strftime('%d %b %Y').upper() for d in dates] + [footing]
    res = '\n'.join(res) + '\n\n'
    return res

def read_restartdate_from_buffer(buffer, attr, logger):
    """Read RESTARTDATE keyword."""
    _, __ = attr, logger
    columns = ['NAME', 'STEP', 'DAY', 'MONTH', 'YEAR']
    full = None
    for line in buffer:
        vals = line.split()[:len(columns)]
        if not line.split('/')[0].strip():
            break
        full = [None] * len(columns)
        shift = 0
        full = parse_vals(columns, shift, full, vals)
    return full

def read_restart_from_buffer(buffer, attr, logger):
    """Read RESTART keyword."""
    _, __ = attr, logger
    columns = ['NAME', 'STEP']
    full = None
    for line in buffer:
        vals = line.split()[:len(columns)]
        if not line.split('/')[0].strip():
            break
        full = [None] * len(columns)
        shift = 0
        full = parse_vals(columns, shift, full, vals)
    return full

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
            shift += int(v.strip('*')) - 1
        else:
            full[i+shift] = v
    return full

def parse_eclipse_keyword(buffer, columns, column_types, defaults=None, date=None):
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
    df = pd.DataFrame(columns=columns)
    for line in buffer:
        if '/' not in line:
            break
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = shlex.split(line)[:len(columns)]
        full = [None] * len(columns)
        if date is not None:
            full[0] = date
            shift = 1
        else:
            shift = 0
        full = parse_vals(columns, shift, full, vals)
        df = pd.concat([df, pd.DataFrame(dict(zip(columns, full)), index=[0])], ignore_index=True)

    if 'text' in column_types:
        df[column_types['text']] = df[column_types['text']].map(
            lambda x: x.strip('\'\"') if x is not None else x)
    if 'float' in column_types:
        df[column_types['float']] = df[column_types['float']].astype(float, errors='ignore')
    if 'int' in column_types:
        df[column_types['int']] = df[column_types['int']].fillna(INT_NAN).astype(int)
    if defaults:
        for k, v in defaults.items():
            if k in df:
                df[k] = df[k].fillna(v)
    return df

def _dummy_loader(buffer):
    """Dummy loader. Read until empty line with `/`. """
    buffer.skip_to('/')

def _dummy_loader2(buffer):
    """Dummy loader. Read until first `/`. """
    for line in buffer:
        if '/' in line:
            break

def _dummy_loader3(buffer):
    """Dummy loader. Read until empty line with `/` after the line ended with `/`. """
    end_line = False
    for line in buffer:
        if line.startswith('/'):
            if end_line:
                break
        end_line = '/' in line
