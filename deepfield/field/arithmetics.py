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
    for _, row in df.iterrows():
        box = _get_box(row[columns[2:]].values, field.grid.dimens)
        copy_successfull = False
        for name, comp in field.items():
            if isinstance(comp, SpatialComponent) and row.ATTR1 in  comp.attributes:
                if row.ATTR2 in comp.state.binary_attributes:
                    copy_successfull = True
                    if logger:
                        logger.info(f'Copy {name}:{row.ATTR1} to {name}:{row.ATTR2} was not applied. ' +
                                    f'{name}:{row.ATTR2} was loaded from binary' )
                    break
                if (box is not None) and row.ATTR2 not in  comp.attributes:
                    default_value, dtype = DEFAULTS_DICT[row.ATTR1]
                    comp.equals_attribute(row.ATTR2, default_value, box=None, dtype=dtype, create=True)
                    if logger:
                        logger.warning(f'Create attribute {name}:{row.ATTR2}')
                comp.copy_attribute(row.ATTR1, row.ATTR2, box)
                if logger is not None:
                    logger.info(f'Copy {name}:{row.ATTR1} to {name}:{row.ATTR2}' +
                                (f' in box {box}' if box is not None else ''))
                copy_successfull = True
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
    for _, row in df.iterrows():
        box = _get_box(row[columns[2:]].values, field.grid.dimens)
        multiply_successfull = False
        for name, comp in field.items():
            if isinstance(comp, SpatialComponent) and row.ATTR in  comp.attributes:
                if row.ATTR in comp.state.binary_attributes:
                    multiply_successfull = True
                    if logger:
                        logger.info(f'Multiply was not applied. {name}:{row.ATTR} was loaded from binary' )
                    break
                comp.multiply_attribute(row.ATTR, row.MULT, box)
                if logger is not None:
                    logger.info(f'Multiply {name}:{row.ATTR} by {row.MULT} in box {box}')
                multiply_successfull = True
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
    for _, row in df.iterrows():
        box = _get_box(row[columns[2:]].values, field.grid.dimens)
        component_found = False
        for name, comp in field.items():
            if isinstance(comp, SpatialComponent) and row.ATTR in comp.attributes:
                component_found = True
                break
        if not component_found:
            for name, attributes in ATTRIBUTES_DICT.items():
                if name not in field.components:
                    continue
                comp = getattr(field, name)
                if row.ATTR in attributes:
                    default_value, dtype = DEFAULTS_DICT[row.ATTR]
                    comp.equals_attribute(row.ATTR, default_value, box=None, dtype=dtype, create=True)
                    if logger:
                        logger.info(f'Create attribute {name}:{row.ATTR}')
                    component_found = True
                    break
        if component_found:
            if row.ATTR in comp.state.binary_attributes:
                if logger:
                    logger.info(f'Equals was not applied. {name}:{row.ATTR} was loaded from binary' )
            comp.equals_attribute(row.ATTR, row.VAL, box)
            if logger:
                logger.info(f'Set {name}:{row.ATTR} to {row.VAL} in box {box}')
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
    for _, row in df.iterrows():
        box = _get_box(row[columns[2:]].values, field.grid.dimens)
        addition_successfull = False
        for name, comp in field.items():
            if isinstance(comp, SpatialComponent) and row.ATTR in comp.attributes:
                if row.ATTR in comp.state.binary_attributes:
                    if logger:
                        logger.info(f'ADD was not applied. {name}:{row.ATTR} was loaded from binary' )
                    addition_successfull = True
                    break
                comp.add_to_attribute(row.ATTR, row.ADDITION, box)
                if logger is not None:
                    logger.info(f'ADD {row.ADDITION} to {name}:{row.ATTR} in box {box}')
                addition_successfull = True
                break
        if not addition_successfull and logger is not None:
            logger.warning(f'Could not find attribute {row.ATTR}')
    return field

def _get_box(vals, dimens):
    full_box = np.array([1, dimens[0], 1, dimens[1], 1, dimens[2]])
    mask = vals == INT_NAN
    if mask.any():
        vals[mask] = full_box[mask]
    vals[[0, 2, 4]] -= 1
    return vals
