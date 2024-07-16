"""Load arithmetics."""
import numpy as np
from .parse_utils.ascii import parse_eclipse_keyword

def _load_table(faults, attribute, columns, column_types, buffer, **kwargs):
    _ = kwargs
    df = parse_eclipse_keyword(buffer, columns, column_types)
    if not df.empty:
        faultsdata = {k: {attribute : v.reset_index(drop=True)} for k, v in df.groupby('NAME')}
        faults.update(faultsdata, mode='a', ignore_index=True)
    return faults

def load_faults(faults, buffer, **kwargs):
    """Partial load FAULTS table."""
    columns = ['NAME', 'IX1', 'IX2', 'IY1', 'IY2', 'IZ1', 'IZ2', 'FACE']
    column_types = {
        'text': [columns[0], columns[-1]],
        'int': columns[1:-1]
    }
    attribute = 'FAULTS'
    return _load_table(faults, attribute, columns, column_types, buffer, **kwargs)

def load_multflt(faults, buffer, **kwargs):
    """Partial load MULTFLT table."""
    columns = ['NAME', 'MULT']
    column_types = {
        'text': [columns[0], columns[-1]],
        'int': columns[1:-1]
    }
    attribute = 'MULTFLT'
    return _load_table(faults, attribute, columns, column_types, buffer, **kwargs)