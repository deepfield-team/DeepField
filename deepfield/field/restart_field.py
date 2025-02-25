"""RestartField class."""
import copy

import pandas as pd
import numpy as np

from .configs import default_config_restart
from .field import Field
from .parse_utils import read_restartdate_from_buffer, tnav_ascii_parser
from .parse_utils.ascii import _get_path, read_restart_from_buffer
from .states import States

class RestartField(Field):
    """
    Restart Reservoir Model.

    Incorporate data and provide routines to work with restaart models.

    Parameters
    ----------
    path : str, optional
        Path to source model files.
    config : dict, optional
        Components and attributes to load.
    logfile : str, optional
        Path to log file.
    encoding : str, optional
        Files encoding. Set 'auto' to infer encoding from initial file block.
        Sometimes it might help to specify block size, e.g. 'auto:3000' will
        read first 3000 bytes to infer encoding.
    loglevel : str, optional
            Log level to be printed while loading. Default to 'INFO'.
    """

    _default_config = default_config_restart

    def __init__(self, path=None, config=None, logfile=None, encoding='auto', loglevel='INFO'):
        super().__init__(path, config, logfile, encoding, loglevel)
        self.parent = None

    def _init_components(self, config):
        if config is None:
            config = self._default_config

        config_tmp = {k.lower(): self._config_parser(v) for k, v in config.items() if k.lower()!='parentmodel'}
        config_tmp['ParentModel'] = config['ParentModel']
        config = config_tmp
        parent_model_config = config['ParentModel']
        restart_config = copy.deepcopy(config)
        del restart_config['ParentModel']
        super()._init_components(restart_config)
        self._restart_config = restart_config
        self._config = config
        self._parent_model_config = parent_model_config

    @property
    def restart_date(self):
        "Show restart date."
        restart_date = self.meta['RESTARTDATE']
        return pd.to_datetime(
            f'{restart_date[4]}-{restart_date[3]}'+
            f'{restart_date[2]}'
        )

    @property
    def grid(self):
        assert self.parent is not None
        return self.parent.grid

    @property
    def restart_path(self):
        """Name of parent model.
        """
        if 'RESTART' in self._meta:
            return self._meta['RESTART'][0]
        return self._meta['RESTARTDATE'][0]


    def _load_data(self, raise_errors=False, include_binary=True):
        loaders = self._get_loaders({})
        tnav_ascii_parser(self._path, loaders, self._logger, encoding=self._encoding,
                          raise_errors=raise_errors)
        assert self._path is not None
        data_dir = self._path.parent
        restart_path = self.restart_path.strip('"\'')+'.data'
        parent_path = _get_path(restart_path, data_dir, self._logger, raise_errors)
        self.parent = Field(str(parent_path),
                            config=self._parent_model_config).load()
        self.wells = self.parent.wells.copy()
        loaders = self._get_loaders(self._restart_config)
        tnav_ascii_parser(self._path, loaders, self._logger, encoding=self._encoding,
                          raise_errors=raise_errors)
        if include_binary:
            self._load_binary(components=('states', 'wells'), raise_errors=raise_errors)
        self._load_results(self._config, raise_errors, include_binary)
        self._check_vapoil(self._config)
        return self

    def _read_buffer(self, buffer, attr, logger):
        if attr not in ('RESTARTDATE', 'RESTART'):
            return super()._read_buffer(buffer, attr, logger)
        if attr == 'RESTARTDATE':
            self.meta['RESTARTDATE'] = read_restartdate_from_buffer(buffer, attr, logger)
        else:
            self.meta['RESTART'] = read_restart_from_buffer(buffer, attr, logger)
        return self

    def full_model(self):
        """Concatenate History Model and Restart Model.

        Returns
        -------
        model: Field
        Concatenated model.
        """
        tmp_model = Field()
        assert self.parent is not None
        for comp in self.parent.components:
            if comp in ('grid', 'rock', 'tables', 'aquifers'):
                setattr(tmp_model, comp, getattr(self.parent, comp).copy())
        result_dates_parent = self.parent.result_dates
        result_dates_restart = self.result_dates
        states_tmp = States()
        for attr in self.states.attributes:
            setattr(states_tmp, attr,
                np.concatenate(
                    (getattr(self.parent.states, attr)[result_dates_parent < result_dates_restart[0]],
                     getattr(self.states, attr)))
            )
        tmp_model.states = states_tmp
        wells_tmp = self.parent.wells
        for node in wells_tmp:
            if node.is_main_branch:
                for attr in node.attributes:
                    if attr in ('EVENTS', 'RESULTS', 'PERF', 'COMPDAT', 'WCONPROD', 'WCONINJE'):
                        setattr(node, attr,
                                pd.concat((
                                    getattr(node, attr)[getattr(node, attr)['DATE'] < result_dates_restart[0]],
                                    getattr(self.wells[node.name], attr)
                                ), ignore_index=True))
        tmp_model.wells = wells_tmp
        tmp_model.meta.update(self.parent.meta)
        tmp_model.meta['Dates'] = self.parent.meta['DATES'].append(self.meta['DATES'])
        if self.state.spatial:
            tmp_model.to_spatial()
        return tmp_model
