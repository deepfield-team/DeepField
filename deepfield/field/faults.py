import numpy as np
import pandas as pd

from .base_component import BaseComponent
from .parse_utils import parse_eclipse_keyword

class Faults(BaseComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
    
    def __delitem__(self, key):
        return super().__delitem__(key)
    
    def __contains__(self, x):
        return super().__contains__(x)
    
    def _read_buffer(self, buffer, attr, **kwargs):
        columns = ['NAME', 'IX1', 'IX2', 'IY1', 'IY2', 'IZ1', 'IZ2', 'FACE']
        column_types = {
        'text': [columns[0], columns[-1]],
        'int': columns[1:-1]
        }
        table = parse_eclipse_keyword(buffer, columns, column_types) #read_table(buffer, TABLE_INFO[attr], dtype)
        setattr(self, attr, table)
        return self