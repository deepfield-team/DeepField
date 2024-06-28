import numpy as np
import pandas as pd

from .base_component import BaseComponent
from .parse_utils import parse_eclipse_keyword


class Faults(BaseComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data = {}
        self._names = []

    def __getitem__(self, key):
        fault = _Fault(self, key)
        return fault

    @property
    def names(self):
        if hasattr(self, 'FAULTS'):
            self._names = self._data['FAULTS']['NAME'].unique()
        return self._names

    def _read_buffer(self, buffer, attr, **kwargs):
        if attr == 'FAULTS':
           columns = ['NAME', 'IX1', 'IX2', 'IY1', 'IY2', 'IZ1', 'IZ2', 'FACE']
           column_types = {
                'text': [columns[0], columns[-1]],
                'int': columns[1:-1]
           }
           table = parse_eclipse_keyword(buffer, columns, column_types)
           self._data['FAULTS'] = table
        elif attr == 'MULTFLT':
            columns = ['NAME', 'MULT']
            column_types = {
                'text': [columns[0], columns[-1]],
                'int': columns[1:-1]
            }
            table = parse_eclipse_keyword(buffer, columns, column_types)
            self._data['MULTFLT'] = table
        return self

class _Fault:
    def __init__(self, faults, key):
        self._faults = faults
        self._key = key

    def __getattr__(self, attr):
        if attr.upper() in self._faults._data:
            df = self._faults._data[attr.upper()]
            return df[df['NAME'] == self._key]
        raise AttributeError("{} has no attribute {}".format(self._key, attr))
