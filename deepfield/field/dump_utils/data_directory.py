from dataclasses import dataclass
from enum import Enum, auto
import os
from typing import Callable, Optional

import pandas as pd
import numpy as np

MAX_STRLEN = 40

FLUID_KEYWORDS = ('OIL', 'GAS', 'WATER')
ORTHOGONAL_GRID_KEYWORDS = ('DX', 'DY', 'DZ', 'TOPS', 'ACTNUM')
ROCK_GRID_KEYWORDS = ('PORO', 'PERMX', 'PERMY', 'PERMZ', 'NTG')

class DataTypes(Enum):
    STRING = auto()
    DATE = auto()
    VECTOR = auto()
    NUMBER = auto()
    STATEMENT_LIST = auto()
    ARRAY = auto()

TABLE_COLUMNS = {
    'RUNCTRL': ('Parameter', 'Value'),
    'TNAVCTRL': ('Parameter', 'Value')
}

@dataclass
class DirectoryEntrySpecification:
    keyword: str
    data_type: Optional[DataTypes]

DATA_DIRECTORY = {
    "RUNSPEC": [
        DirectoryEntrySpecification('TITLE', DataTypes.STRING),
        DirectoryEntrySpecification('MULTOUT', None),
        DirectoryEntrySpecification('MULTOUTS', None),
        DirectoryEntrySpecification('START', DataTypes.STRING),
        DirectoryEntrySpecification('METRIC', None),
        *[DirectoryEntrySpecification( fluid, None) for fluid in FLUID_KEYWORDS],
        DirectoryEntrySpecification('DIMENS', DataTypes.VECTOR),
        DirectoryEntrySpecification('RUNCTRL', DataTypes.STATEMENT_LIST),
        DirectoryEntrySpecification('TNAVCTRL', DataTypes.STATEMENT_LIST)
],
    "GRID": [
        DirectoryEntrySpecification('MAPAXES', DataTypes.VECTOR),
        *[DirectoryEntrySpecification(keyword, DataTypes.ARRAY) for keyword in ORTHOGONAL_GRID_KEYWORDS],
        *[DirectoryEntrySpecification( keyword, DataTypes.ARRAY) for keyword in ROCK_GRID_KEYWORDS]
    ]
}

def dump_keyword(spec, val, buf, include_path):
    _DUMP_ROUTINES[spec.data_type](spec.keyword, val, buf, include_path)
    buf.write('\n')
    return buf

def _dump_array(keyword, val, buf, include_dir):
    buf.write(keyword+'\n')
    with open(os.path.join(include_dir, f'{keyword}.inc'), 'w') as inc_buf:
        dump_array_ascii(inc_buf, val.reshape(-1), fmt='%.3f')
    buf.write('\t'.join(('INCLUDE', f'"{os.path.join(os.path.split(include_dir)[1], f"{keyword}.inc")}"')))
    buf.write('\n/\n')

_DUMP_ROUTINES = {
    DataTypes.STRING: lambda keyword, val, buf, _: buf.write('\n'.join([keyword, val, '/\n'])),
    DataTypes.STATEMENT_LIST: lambda keyword, val, buf, _: buf.write('\n'.join([keyword] +
       ['\t'.join([str(value) for value in row[1].values.tolist() + ['/']]) for row in val.iterrows()] +
        ['/\n']
    )),
    DataTypes.VECTOR: lambda keyword, val, buf, _: buf.write('\n'.join([keyword, '\t'.join(map(str, val)), '/\n'])),
    DataTypes.ARRAY: _dump_array,
    None: lambda keyword, _, buf, ___: buf.write(f'{keyword}\n')
}

def dump_array_ascii(buffer, array, header=None, fmt='%f', compressed=True):
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
