"""Load arithmetics."""
import itertools

import numpy as np
from .parse_utils.ascii import INT_NAN, parse_eclipse_keyword
from .base_spatial import SpatialComponent

ATTRIBUTES_DICT = {
    'grid': ('DX', 'DY', 'DZ', 'TOPS'),
    'rock': ('PERMX', 'PERMY', 'PERMZ', 'MULTX', 'MULTY', 'MULTZ', 'PORO', 'SWL',
             'SWU', 'SWCR', 'SGL', 'SGCR', 'SGU', 'KRW', 'KRO', 'KRGR', 'SOWCR', 'SOGCR'),
}
DEFAULTS_DICT = {key: (0, float) for key in itertools.chain(*(val for val in ATTRIBUTES_DICT.values()))}

def load_copy(field, buffer, logger=None):
    """Load COPY keyword"""
    columns = ['ATTR1', 'ATTR2',  'I1', 'I2', 'J1', 'J2', 'K1', 'K2']
    column_types = {
        'text': columns[:2],
        'int': columns[2:]
    }
    df = parse_eclipse_keyword(buffer, columns, column_types)
    for row in df.itertuples():
        box = _get_box(getattr(row, c) for c in columns[2:])
        copy_successfull = False
        for comp in field.components:
            if isinstance(getattr(field, comp), SpatialComponent) and row.ATTR1 in  getattr(field, comp).attributes:
                if row.ATTR2 not in comp.state.binary_attributes:
                    if (box is not None) and row.ATTR2 not in  getattr(field, comp).attributes:
                        default_value, dtype = DEFAULTS_DICT[row.ATTR1]
                        getattr(field, comp).equals_attribute(row.ATTR2, default_value, box=None, dtype=dtype, create=True)
                        if logger:
                            logger.warning(f'Create attribute {comp}:{row.ATTR2}')
                    getattr(field, comp).copy_attribute(row.ATTR1, row.ATTR2, box)
                    if logger is not None:
                        logger.info(f'Copy {comp}:{row.ATTR1} to {comp}:{row.ATTR2}' +
                                    (f' in box {box}' if box is not None else ''))
                    copy_successfull = True
                else:
                    copy_successfull = True
                    if logger:
                        logger.info(f'Copy {comp}:{row.ATTR1} to {comp}:{row.ATTR2} was not applied. {comp}:{row.ATTR2} was loaded from binary' )
                break
        if not copy_successfull and logger is not None:
            logger.warning(f'Could not find attribute {row.ATTR1}')
    return field

def load_multiply(field, buffer, logger=None):
    """Load MULTIPLY keyword."""
    columns = ['ATTR', 'MULT', 'I1', 'I2', 'J1', 'J2', 'K1', 'K2']
    column_types = {
        'text': columns[0:1],
        'int': columns[2:],
        'float': columns[1:2]
    }
    df = parse_eclipse_keyword(buffer, columns, column_types)
    for row in df.itertuples():
        box = _get_box(getattr(row, c) for c in columns[2:])
        multiply_successfull = False
        for comp in field.components:
            if isinstance(getattr(field, comp), SpatialComponent) and row.ATTR in  getattr(field, comp).attributes:
                if row.ATTR not in comp.state.binary_attributes:
                    getattr(field, comp).multiply_attribute(row.ATTR, row.MULT, box)
                    if logger is not None:
                        logger.info(f'Multiply {comp}:{row.ATTR} by {row.MULT} in box {box}')
                    multiply_successfull = True
                else:
                    multiply_successfull = True
                    if logger:
                        logger.info(f'Multiply was not applied. {comp}:{row.ATTR} was loaded from binary' )
                break
        if not multiply_successfull and logger is not None:
            logger.warning(f'Could not find attribute {row.ATTR}')
    return field

def load_equals(field, buffer, logger=None):
    "Load EQUALS keyword."
    columns = ['ATTR', 'VAL', 'I1', 'I2', 'J1', 'J2', 'K1', 'K2']
    column_types = {
        'text': columns[0:1],
        'int': columns[2:],
        'float': columns[1:2]
    }
    df = parse_eclipse_keyword(buffer, columns, column_types)
    for row in df.itertuples():
        box = _get_box(getattr(row, c) for c in columns[2:])
        component_found = False
        for comp in field.components:
            if isinstance(getattr(field, comp), SpatialComponent) and row.ATTR in  getattr(field, comp).attributes:
                component_found = True
                break
        if not component_found:
            for comp, attributes in ATTRIBUTES_DICT.items():
                if row.ATTR in attributes:
                    default_value, dtype = DEFAULTS_DICT[row.ATTR]
                    getattr(field, comp).equals_attribute(row.ATTR, default_value, box=None, dtype=dtype, create=True)
                    if logger:
                        logger.warning(f'Create attribute {comp}:{row.ATTR}')
                    component_found = True
                    break
        if component_found:
            if row.ATTR not in getattr(field, comp).state.binary_attributes:
                getattr(field, comp).equals_attribute(row.ATTR, row.VAL, box)
                if logger:
                    logger.info(f'Set {comp}:{row.ATTR} to {row.VAL} in box {box}')
            else:
                if logger:
                    logger.info(f'Equals was not applied. {comp}:{row.ATTR} was loaded from binary' )
        else:
            if logger:
                logger.warning(f'Could not find or create attribute {row.ATTR}')
    return field

def load_add(field, buffer, logger=None):
    """Load ADD keyword."""
    columns = ['ATTR', 'ADDITION', 'I1', 'I2', 'J1', 'J2', 'K1', 'K2']
    column_types = {
        'text': columns[0:1],
        'int': columns[2:],
        'float': columns[1:2]
    }
    df = parse_eclipse_keyword(buffer, columns, column_types)
    for row in df.itertuples():
        box = _get_box(getattr(row, c) for c in columns[2:])
        addition_successfull = False
        for comp in field.components:
            if isinstance(getattr(field, comp), SpatialComponent) and row.ATTR in  getattr(field, comp).attributes:
                if row.ATTR not in getattr(field, comp).state.binary_attributes:
                    getattr(field, comp).add_to_attribute(row.ATTR, row.ADDITION, box)
                    if logger is not None:
                        logger.info(f'ADD {row.ADDITION} to {comp}:{row.ATTR} in box {box}')
                    addition_successfull = True
                else:
                    if logger:
                        logger.info(f'ADD was not applied. {comp}:{row.ATTR} was loaded from binary' )
                break
        if not addition_successfull and logger is not None:
            logger.warning(f'Could not find attribute {row.ATTR}')
    return field

def _get_box(vals):
    vals = list(vals)
    if (np.array(vals) == INT_NAN).all():
        return None
    if (np.array(vals) == INT_NAN).any() or len(vals) != 6:
        raise ValueError('Box is not properly specified')
    return [val-1 if i%2==0 else val for i, val in enumerate(vals)]
