import copy
import numpy as np
from .data_directory import INT_NAN, DataTypes, DATA_DIRECTORY

def dump_keyword(keyword, val, section,  buf, include_path):
    DUMP_ROUTINES[DATA_DIRECTORY[section][keyword]](keyword, val, buf, include_path)
    buf.write('\n')
    return buf

def _dump_array(keyword, val, buf, include_dir):
    buf.write(keyword+'\n')
    with open(os.path.join(include_dir, f'{keyword}.inc'), 'w') as inc_buf:
        _dump_array_ascii(inc_buf, val.reshape(-1), fmt='%.3f')
    buf.write('\t'.join(('INCLUDE', f'"{os.path.join(os.path.split(include_dir)[1], f"{keyword}.inc")}"')))
    buf.write('\n/\n')

def _dump_table(keyword, val, buf):
    buf.write(keyword + '\n')
    for table in val:
        for _, row in table.iterrows():
            buf.write('\t'.join([str(v) for v in row.values] + ['\n']))
        buf.write('/\n')

def _dump_single_statement(keyword, val, buf):
    buf.write(keyword + '\n')
    _dump_statement(val, buf, closing_slash=False)
    buf.write('/\n')

DUMP_ROUTINES = {
    DataTypes.STRING: lambda keyword, val, buf, _: buf.write('\n'.join([keyword, val, '/\n'])),
    DataTypes.STATEMENT_LIST: lambda keyword, val, buf, _: buf.write('\n'.join([keyword] +
       ['\t'.join([str(value) for value in row[1].values.tolist() + ['/']]) for row in val.iterrows()] +
        ['/\n']
)),
    DataTypes.ARRAY: _dump_array,
    DataTypes.TABLE_SET: lambda keyword, val, buf, _: _dump_table(keyword, val, buf),
    None: lambda keyword, _, buf, ___: buf.write(f'{keyword}\n'),
    DataTypes.SINGLE_STATEMENT: _dump_single_statement
}

def _dump_statement(val, buf, closing_slash=True):
    vals = val.values
    if vals.shape[0] != 1:
        raise ValueError('Val shoud have exactly one row.')
    vals = vals.reshape(-1)

    vals = [nan_to_none(v) for v in vals]
    print(vals)
    str_representaions = [str(v) if v is not None else '' for v in vals]
    str_representaions = _replace_empty_vals(str_representaions)
    result = '\t'.join(str_representaions)
    result += '\n' if not closing_slash else '/\n'
    buf.write(result)

def _replace_empty_vals(vals):
    vals = copy.copy(vals)
    print(vals)
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
    if val == np.NaN:
        return None
    if val == INT_NAN:
        return None
    if val == '':
        return None
    return val

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
            else:
                buffer.write(' ')
        buffer.write('\n')
    else:
        for i in range(0, len(array), MAX_STRLEN):
            buffer.write(' '.join([fmt % d for d in array[i:i + MAX_STRLEN]]))
            buffer.write('\n')
        buffer.write('\n')
