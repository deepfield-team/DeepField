import pandas as pd
from .dump_utils.data_directory import FLUID_KEYWORDS, ORTHOGONAL_GRID_KEYWORDS, ROCK_GRID_KEYWORDS, TABLE_COLUMNS

DATA_GETTERS = {
    "RUNSPEC": {
        'TITLE':  lambda field: _meta_getter('TITLE', field),
        'MULTOUT': lambda field: (True, None),
        'MULTOUTS': lambda field: (True, None),
        'START':  lambda field: _meta_getter('START', field),
        'METRIC': lambda field: _units_getter('METRIC', field),
        'FIELD': lambda field: _units_getter('FIELD', field),
        **{fluid: lambda field, fluid=fluid: _fluids_getter(fluid, field) for fluid in FLUID_KEYWORDS},
        'DIMENS':  lambda field: (True, field.grid.DIMENS),
        'RUNCTRL': lambda field: (True, pd.DataFrame(
            [['WELLEQUATIONS', 1], ['WATERZONE', 1]],
            columns=TABLE_COLUMNS['RUNCTRL']
        )),
        'TNAVCTRL': lambda field: (True, pd.DataFrame(
            [['LONGNAMES', 1]],
            columns=TABLE_COLUMNS['TNAVCTRL']
        )),
    },
    "GRID": {
        'MAPAXES':  lambda field: field.grid.MAPAXES,
        **{keyword:  lambda field, keyword=keyword: _array_getter(keyword, field.grid)
                                                    for keyword in ORTHOGONAL_GRID_KEYWORDS},
        **{keyword:  lambda field, keyword=keyword: _array_getter(keyword, field.states)
                                                    for keyword in ROCK_GRID_KEYWORDS}
    }
}

def _meta_getter(keyword, field):
    return True, field.meta[keyword]

def _units_getter(keyword, field):
    units = field.meta.get('UNITS', 'METRIC')
    return keyword == units, None

def _fluids_getter(keyword, field):
    return keyword in field.meta['FLUIDS'], None

def _array_getter(keyword, component):
    is_present = keyword in component.attributes
    val = getattr(component, keyword).reshape(-1, order='F') if is_present else None
    return is_present, val
