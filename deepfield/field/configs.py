# pylint: skip-file
"""Configs collection."""
from .parse_utils.table_info import TABLE_INFO

orth_grid_config = {'OrthogonalGrid': {'attrs': ['ACTNUM', 'DIMENS', 'DX', 'DY', 'DZ',
                                                 'DXV', 'DYV', 'DZV',
                                                 'MAPAXES', 'TOPS', 'MINPV']}}
corn_grid_config = {'CornerPointGrid': {'attrs': ['ACTNUM', 'COORD', 'DIMENS', 'MAPAXES', 'ZCORN', 'MINPV'],
                                        'apply_mapaxes': True}}
any_grid_config = {'Grid': {'attrs': list(set(orth_grid_config['OrthogonalGrid']['attrs'] +
                                              corn_grid_config['CornerPointGrid']['attrs']))}}

base_config = {
    'Rock': {'attrs': ['PERMX', 'PERMY', 'PERMZ', 'PORO', 'KRW', 'KRWR', 'SGU', 'SOGCR', 'SOWCR', 'SWATINIT', 'SWCR', 'SWL', 'NTG']},
    'States': {'attrs': ['PRESSURE', 'SOIL', 'SWAT', 'SGAS', 'RS']},
    'Tables': {'attrs': list(TABLE_INFO.keys())},
    'Wells': {'attrs': ['EVENTS', 'HISTORY', 'RESULTS', 'PERF', 'WELLTRACK',
                        'COMPDAT', 'WELSPECS', 'WELSPECL', 'WCONPROD', 'WCONINJE',
                        'COMPDATL', 'COMPDATMD', 'WEFAC', 'WFRAC', 'WFRACP']},
    'Faults': {'attrs': ['FAULTS', 'MULTFLT']},
    'Aquifers': {'attrs': None}
}

default_orth_config = dict(orth_grid_config, **base_config)
default_corn_config = dict(corn_grid_config, **base_config)
default_config = dict(any_grid_config, **base_config)

def _restart(config):
    return dict(States=config['States'], Wells=config['Wells'], ParentModel=config)

(default_config_restart,
 default_orth_config_restart,
 default_corn_config_restart) = map(_restart, (default_config, default_orth_config, default_corn_config))