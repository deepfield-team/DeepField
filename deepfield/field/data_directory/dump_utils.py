from contextlib import ExitStack
import copy
import numbers
import numpy as np
import pandas as pd
from .data_directory import INT_NAN, DataTypes, DATA_DIRECTORY, get_dynamic_keyword_specification

MAX_STRLEN = 40

INPLACE_ARRAYS = ['TSTEP']

def format_string_val(val, keyword_spec):
    if keyword_spec.specification is not None and keyword_spec.specification.date:
        d = val.strftime('%d %b %Y').upper()
        if val.hour or val.minute or val.second:
            t = val.strftime('%H:%M:%S')
            return ' '.join((d, t))
        return d
    return val

def _dump_string(keyword_spec, val, buf):
     buf.write('\n'.join([keyword_spec.keyword, format_string_val(val, keyword_spec), '/']))

def dump_keyword(keyword, val, section,  buf, include_path):
    DUMP_ROUTINES[DATA_DIRECTORY[section][keyword]](keyword, val, buf, include_path)
    buf.write('\n')
    return buf

def _dump_array(keyword_spec, val, buf, include_dir):
    if keyword_spec.keyword in INPLACE_ARRAYS:
        inplace = True
    else:
        inplace = False
    if keyword_spec.specification.dtype in (bool, int):
        fmt = '%d'
    else:
        fmt = '%f'
    if inplace:
        _dump_array_ascii(buf, val.reshape(-1), header=keyword_spec.keyword, fmt=fmt)
        buf.write('/')
        return
    with open(include_dir/f'{keyword_spec.keyword}.inc', 'w') as inc_buf:
        _dump_array_ascii(inc_buf, val.reshape(-1), fmt=fmt, header=keyword_spec.keyword)
        inc_buf.write('/')
    buf.write('\n'.join(('INCLUDE', '"' + '/'.join((include_dir.name, f'{keyword_spec.keyword}.inc')) + '"')))
    buf.write('\n/')

def _dump_table(keyword_spec, val, buf):
    buf.write(keyword_spec.keyword)
    domain = keyword_spec.specification.domain
    for table in val:
        if keyword_spec.specification.domain is not None and len(keyword_spec.specification.domain) == 2:
            _dump_multitable(table, buf)
            continue
        buf.write('\n')
        row_iterator = (table.itertuples() if domain is not None else table.itertuples(index=False))
        for row in row_iterator:
            vals = list(row)
            vals = [nan_to_none(v) for v in vals]
            str_representaions = [_string_representation(v) if v is not None else '' for v in vals]
            str_representaions = _replace_empty_vals(str_representaions)
            buf.write('\t'.join([v for v in str_representaions]) + '\n')
        buf.write('/')

def _dump_multitable(val, buf):
    buf.write('\n')
    for ind0, df in val.groupby(level=0):
        for i, (ind1, row) in enumerate(df.iterrows()):
            vals = row.values.tolist()
            if i == 0:
                vals =  [*ind1] + vals
            else:
                vals = [ind1[1]] + vals
            vals = [nan_to_none(v) for v in vals]
            str_representations = [_string_representation(v) if v is not None else '' for v in vals]
            str_representations = _replace_empty_vals(str_representations)
            if i !=0:
                str_representations = [''] + str_representations
            if i == len(df) - 1:
                str_representations = str_representations + ['/']
            buf.write('\t'.join(str_representations) + '\n')
    buf.write('/')


def _dump_single_statement(keyword_spec, val, buf):
    buf.write(keyword_spec.keyword + '\n')
    _dump_statement(val, buf, closing_slash=False)
    buf.write('/')

def _dump_statement_list(keyword_spec, val, buf):
    buf.write(keyword_spec.keyword + '\n')
    for row in val.itertuples(index=False):
        _dump_statement(row, buf, closing_slash=True)
    buf.write('/')

def _dump_records(keyword_spec, val, buf):
    buf.write(keyword_spec.keyword + '\n')
    for v in val:
        _dump_statement(v, buf, closing_slash=True)

def _dump_object_list(keyword_spec, val, buf):
    buf.write(keyword_spec.keyword + '\n')
    for o in val:
        buf.write(f'{format_string_val(o, keyword_spec)}')
        if keyword_spec.specification is not None and keyword_spec.specification.terminated:
            buf.write(' /')
        buf.write('\n')
    buf.write('/')

def dump_parameters(keyword_spec, val, buf):
    if keyword_spec.specification.tabulated:
        return dump_tabulated_parameters(keyword_spec, val, buf)

    buf.write(keyword_spec.keyword + '\n')
    res = ' '.join(
        [f'{k}' if v is None else f'{k}={v}' for k, v in val.items()]
    )
    buf.write(res)
    buf.write('\n/')

def dump_tabulated_parameters(keyword_spec, val, buf):
    buf.write(keyword_spec.keyword + '\n')
    for key, data in val.items():
        buf.write('\t'.join((key, data)) + '\n')
    buf.write('/')

DUMP_ROUTINES = {
    DataTypes.OBJECT_LIST: lambda keyword_spec, val, buf, _: _dump_object_list(keyword_spec, val, buf),
    DataTypes.STRING: lambda keyword_spec, val, buf, _: _dump_string(keyword_spec, val, buf),
    DataTypes.STATEMENT_LIST: lambda keyword_spec, val, buf, _: _dump_statement_list(keyword_spec, val, buf),
    DataTypes.PARAMETERS: lambda keyword_spec, val, buf, _: dump_parameters(keyword_spec, val, buf),
    DataTypes.ARRAY: _dump_array,
    DataTypes.TABLE_SET: lambda keyword_spec, val, buf, _=None: _dump_table(keyword_spec, val, buf),
    None: lambda keyword_spec, _, buf, ___: buf.write(f'{keyword_spec.keyword}'),
    DataTypes.SINGLE_STATEMENT: lambda keyword_spec, val, buf, _: _dump_single_statement(keyword_spec, val, buf),
    DataTypes.RECORDS: lambda keyword_spec, val, buf, _: _dump_records(keyword_spec, val, buf),
}

def _dump_statement(val, buf, closing_slash=True):
    if isinstance(val, pd.DataFrame):
        if val.shape[0] != 1:
            raise ValueError('Val shoud have exactly one row.')
        vals = [val[col][0] for col in val.columns]
    elif isinstance(val, pd.Series):
        vals = val.values
    else:
        vals = val
    vals = [nan_to_none(v) for v in vals]
    str_representaions = [_string_representation(v) if v is not None else '' for v in vals]
    str_representaions = _replace_empty_vals(str_representaions)
    if len(str_representaions) == 0:
        str_representaions.append('*')
    result = '\t'.join(str_representaions)
    result += '\n' if not closing_slash else '/\n'
    buf.write(result)

def _string_representation(v):
    if v is None:
        return ''
    if isinstance(v, numbers.Number):
        r = str(v)
        if 'e' in r:
            return np.format_float_scientific(v, unique=True, trim='-', exp_digits=1).upper()
        return r
    if isinstance(v, str):
        if any(symbol in v for symbol in ' \t'):
            return "'" + v + "'"
    return str(v)


def _replace_empty_vals(vals):
    vals = copy.copy(vals)
    while True:
        start = None
        end = None
        for i, s in enumerate(vals):
            if s != '':
                if start is not None:
                    break
            else:
                if start is None:
                    start = i
                    end = i
                else:
                    end = i
        if start is None:
            break
        else:
            assert end is not None
            if end == len(vals)-1:
                del vals[start:]
                continue
            if start == end:
                replacement = '*'
            else:
                replacement = f'{end-start+1}*'
            vals[start:end+1] = [replacement]

    return vals

def nan_to_none(val):
    if isinstance(val, numbers.Number) and np.isnan(val):
        return None
    if val == INT_NAN:
        return None
    if val == '':
        return None
    return val

def dump(data, path, inplace_scedule=False, filename=None):
    if not path.exists():
        path.mkdir()

    include_dir = path / 'include'

    if not include_dir.exists():
        include_dir.mkdir()

    if filename is None:
        for key, val in data['RUNSPEC']:
            if key=='TITLE':
                filename = f'{val}.data'

    if filename is None:
        raise ValueError('Filename is not specified and no TITLE keyword in RUNSPEC section.')

    with ExitStack() as stack:
        buf = stack.enter_context(open(path / filename, 'w'))
        for section in ('', 'RUNSPEC', 'GRID', 'EDIT', 'PROPS', 'REGIONS', 'SOLUTION', 'SUMMARY', 'SCHEDULE'):
            if section in data:
                if section != '':
                    buf.write(f'{section}\n\n')
                if section == 'SCHEDULE' and not inplace_scedule:
                    schedule_path = include_dir / 'schedule.inc'
                    buf.write('INCLUDE\n')
                    buf.write(('"' + str(schedule_path.relative_to(path))) + '"')
                    buf.write('\n/\n\n')
                    buf_tmp = stack.enter_context(open(schedule_path, 'w'))
                else:
                    buf_tmp = buf
                for (key, val) in data[section]:
                    if DATA_DIRECTORY[key] is not None:
                        spec = DATA_DIRECTORY[key]
                    else:
                        spec = get_dynamic_keyword_specification(key, data)
                    DUMP_ROUTINES[spec.type](spec, val, buf_tmp, include_dir)
                    buf_tmp.write('\n\n')

def _dump_array_ascii(buffer, array, header=None, fmt='%f', compressed=True):
    """Writes array-like data into an ASCII buffer.

    Parameters
    ----------
    buffer : buffer-like
    array : 1d, array-like
        Array to be saved
    header : str, optional
        String to be written line before the array
    fmt : str or sequence of strs, optional
        Format to be passed into ``numpy.savetxt`` function. Default to '%f'.
    compressed : bool
        If True, uses compressed typing style
    """
    if header is not None:
        buffer.write(header + '\n')

    if compressed:
        i = 0
        items_written = 0
        while i < len(array):
            count = 1
            while (i + count < len(array)) and (array[i + count] == array[i]):
                count += 1
            if count <= 4:
                buffer.write(' '.join([fmt % array[i]] * count))
                items_written += count
            else:
                buffer.write(str(count) + '*' + fmt % array[i])
                items_written += 1
            i += count
            if items_written > MAX_STRLEN:
                buffer.write('\n')
                items_written = 0
            elif i < len(array):
                buffer.write(' ')
        buffer.write('\n')
    else:
        for i in range(0, len(array), MAX_STRLEN):
            buffer.write(' '.join([fmt % d for d in array[i:i + MAX_STRLEN]]))
            buffer.write('\n')
        buffer.write('\n')
