# pylint: disable=too-many-lines
"""Field class."""
import logging
import os
import sys
import weakref
from copy import deepcopy
from functools import partial
from string import Template

import h5py
import numpy as np
import pandas as pd
import pyvista as pv
from anytree import PreOrderIter
from vtk import vtkXMLUnstructuredGridWriter # pylint: disable=no-name-in-module
from vtk.util.numpy_support import numpy_to_vtk # pylint: disable=no-name-in-module, import-error

from .arithmetics import load_add, load_copy, load_equals, load_multiply
from .faults import Faults
from .aquifer import Aquifers
from .configs import default_config
from .dump_ecl_utils import egrid, init, restart, summary
from .grids import CornerPointGrid, Grid, OrthogonalGrid, specify_grid
from .parse_utils import (dates_to_str, preprocess_path,
                          read_dates_from_buffer, tnav_ascii_parser)
from .rates import calc_rates, calc_rates_multiprocess
from .rock import Rock
from .states import States
from .tables import Tables
from .template_models import (CORNERPOINT_GRID, DEFAULT_ECL_MODEL,
                              DEFAULT_TN_MODEL, ORTHOGONAL_GRID)
from .utils import (get_single_path, get_spatial_cf_and_perf,
                    get_spatial_well_control, get_well_mask)
from .wells import Wells

ACTOR = None

COMPONENTS_DICT = {'cornerpointgrid': ['grid', CornerPointGrid],
                   'orthogonalgrid': ['grid', OrthogonalGrid],
                   'grid': ['grid', Grid],
                   'rock': ['rock', Rock],
                   'states': ['states', States],
                   'wells': ['wells', Wells],
                   'tables': ['tables', Tables],
                   'aquifers': ['aquifers', Aquifers],
                   'faults': ['faults', Faults]
                   }

DEFAULT_HUNITS = {'METRIC': ['sm3/day', 'ksm3/day', 'ksm3', 'Msm3', 'bara'],
                  'FIELD': ['stb/day', 'Mscf/day', 'Mstb', 'MMscf', 'psia']}

SECTIONS_DICT = {
    'GRID': [('PORO', 'rock'), ('PERMX', 'rock'), ('PERMY', 'rock'), ('PERMZ', 'rock'), ('MULTZ', 'rock')],
    'PROPS': [('SWATINIT', 'rock'), ('SWL', 'rock'), ('SWCR', 'rock'), ('SGU', 'rock'), ('SGL', 'rock'),
              ('SGCR', 'rock'), ('SOWCR', 'rock'), ('SOGCR', 'rock'), ('SWU', 'rock'), ('ISWCR', 'rock'),
              ('ISGU', 'rock'), ('ISGL', 'rock'), ('ISGCR', 'rock'), ('ISWU', 'rock'), ('ISGU', 'rock'),
              ('ISGL', 'rock'), ('ISWL', 'rock'), ('ISOGCR', 'rock'), ('ISOWCR', 'rock')]
}

SUMMARY_KW = ['WLPR', 'WOPR', 'WGPR', 'WWPR', 'WWIR', 'WGIR', 'WBHP',
              'EXCEL', 'RPTONLY', 'SEPARATE']

META_KW = ['ARRA', 'ARRAY', 'DATES', 'TITLE', 'START', 'METRIC', 'FIELD',
           'HUNI', 'HUNITS', 'OIL', 'GAS', 'WATER', 'DISGAS', 'VAPOIL', 'RES',
           'RESTARTDATE', 'RESTART'] + SUMMARY_KW


#pylint: disable=protected-access
class FieldState:
    """State holder."""
    def __init__(self, field):
        self._field = weakref.ref(field)

    @property
    def field(self):
        """Reference Field."""
        return self._field()


class Field:
    """Reservoir model.

    Contains components of the reservoir model and preprocessing tools.

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
    _default_config = default_config
    def __init__(self, path=None, config=None, logfile=None, encoding='auto', loglevel='INFO'):
        self._path = preprocess_path(path) if path is not None else None
        self._encoding = encoding
        self._components = {}
        self._config = None
        self._meta = {'UNITS': 'METRIC',
                      'START': pd.to_datetime(''),
                      'DATES': pd.to_datetime([]),
                      'FLUIDS': [],
                      'SUMMARY': [],
                      'MODEL_TYPE': '',
                      'HUNITS': DEFAULT_HUNITS['METRIC']}
        self._state = FieldState(self)

        logging.shutdown()
        handlers = [logging.StreamHandler(sys.stdout)]
        if logfile is not None:
            handlers.append(logging.FileHandler(logfile, mode='w'))
        logging.basicConfig(handlers=handlers)
        self._logger = logging.getLogger('Field')
        self._logger.setLevel(getattr(logging, loglevel))

        if self._path is not None:
            self._init_components(config)

        self._pyvista_grid = None
        self._pyvista_grid_params = {'use_only_active': True, 'cell_size': None, 'scaling': True}

    def _init_components(self, config):
        """Initialize components."""
        fmt = self._path.suffix.strip('.').upper()
        if config is not None:
            config = {k.lower(): self._config_parser(v) for k, v in config.items()}
        if fmt == 'HDF5':
            with h5py.File(self._path, 'r') as f:
                keys = [k.lower() for k in f]
                if config is None:
                    config = {k: {'attrs': None, 'kwargs': {}} for k in keys}
                elif 'grid' in config:
                    if 'cornerpointgrid' in keys:
                        config['cornerpointgrid'] = config.pop('grid')
                    elif 'orthogonalgrid' in keys:
                        config['orthogonalgrid'] = config.pop('grid')
        elif config is None:
            self._logger.info('Using default config.')
            config = {k.lower(): self._config_parser(v) for k, v in default_config.items()}

        for k in config:
            setattr(self, COMPONENTS_DICT[k][0], COMPONENTS_DICT[k][1](field=self))
        self._config = {COMPONENTS_DICT[k][0]: v for k, v in config.items()}

    @staticmethod
    def _config_parser(value):
        """Separate config into attrs and kwargs."""
        if isinstance(value, str):
            attrs = [value.upper()]
            kwargs = {}
        elif isinstance(value, (list, tuple)):
            attrs = [x.upper() for x in value]
            kwargs = {}
        elif isinstance(value, dict):
            attrs = value['attrs']
            if attrs is None:
                pass
            elif isinstance(attrs, str):
                attrs = [attrs.upper()]
            else:
                attrs = [x.upper() for x in attrs]
            kwargs = {k: v for k, v in value.items() if k != 'attrs'}
        else:
            raise TypeError("Component's config should be of type str, list, tuple or dict. Found {}."
                            .format(type(value)))
        return {'attrs': attrs, 'kwargs': kwargs}

    @property
    def meta(self):
        """"Model meta data."""
        return self._meta

    @property
    def state(self):
        """"Field state."""
        return self._state

    @property
    def start(self):
        """Model start time in a datetime format."""
        return pd.to_datetime(self.meta['START'])

    @property
    def path(self):
        """Path to original model."""
        if self._path is not None:
            return str(self._path)
        raise ValueError("Model has no file to originate from.")

    @property
    def basename(self):
        """Model filename without extention."""
        fname = os.path.basename(self.path)
        return os.path.splitext(fname)[0]

    @property
    def components(self):
        """Model components."""
        return tuple(self._components.keys())

    def items(self):
        """Returns pairs of components's names and instance."""
        return self._components.items()

    @property
    def grid(self):
        """Grid component."""
        return self._components['grid']

    @grid.setter
    def grid(self, x):
        """Grid component setter."""
        x.field = self
        self._components['grid'] = x
        return self

    @property
    def wells(self):
        """Wells component."""
        return self._components['wells']

    @wells.setter
    def wells(self, x):
        """Wells component setter."""
        x.field = self
        self._components['wells'] = x
        return self

    @property
    def rock(self):
        """Rock component."""
        return self._components['rock']

    @rock.setter
    def rock(self, x):
        """Rock component setter."""
        x.field = self
        self._components['rock'] = x
        return self

    @property
    def states(self):
        """States component."""
        return self._components['states']

    @states.setter
    def states(self, x):
        """States component setter."""
        x.field = self
        self._components['states'] = x
        return self

    @property
    def faults(self):
        """Faults component."""
        return self._components['faults']

    @faults.setter
    def faults(self, x):
        """Faults component setter."""
        x.field = self
        self._components['faults'] = x
        return self

    @property
    def aquifers(self):
        """Aquifers component."""
        return self._components['aquifers']

    @aquifers.setter
    def aquifers(self, x):
        """States component setter."""
        x.field = self
        self._components['aquifers'] = x
        return self

    @property
    def tables(self):
        """Tables component."""
        return self._components['tables']

    @tables.setter
    def tables(self, x):
        """Tables component setter."""
        x.field = self
        self._components['tables'] = x
        return self

    def spatial_cf_and_perf(self, date_range=None, mode=None):
        """Get model's connection factors and perforation ratios in a spatial form.

        Parameters
        ----------
        date_range: tuple
            Minimal and maximal dates for events.
        mode: str, None
            If not None, pick the blocks only with specified mode.

        Returns
        -------
        connection_factors: np.array
        perf_ratio: np.array
        """
        return get_spatial_cf_and_perf(self, date_range=date_range, mode=mode)

    @property
    def result_dates(self):
        """Result dates, actual if present, target otherwise."""
        return self.wells.result_dates

    @property
    def well_mask(self):
        """Get the model's well mask in a spatial form.

        Returns
        -------
        well_mask: np.array
            Array with well-names in cells which are registered as well-blocks and empty strings everywhere else.
        """
        return get_well_mask(self)

    def spatial_well_control(self, attrs, date_range=None, fill_shut=0., fill_outside=0.):
        """Get the model's control in a spatial form.

        Parameters
        ----------
        attrs: tuple or list
            Conrol attributes to get data from.
        date_range: tuple
            Minimal and maximal dates for control events.
        fill_shut: float
            Value to fill shutted perforations.
        fill_outside:
            Value to fill non-perforated cells.

        Returns
        -------
        control: np.array
        """
        return get_spatial_well_control(self, attrs, date_range=date_range,
                                        fill_shut=fill_shut, fill_outside=fill_outside)

    def set_state(self, **kwargs):
        """State setter."""
        for k, v in kwargs.items():
            setattr(self.state, k, v)
        return self

    def copy(self):
        """Returns a deepcopy of Field."""
        copy = self.__class__()
        for k, v in self.items():
            setattr(copy, k, v.copy())
        copy._meta = deepcopy(self.meta) #pylint: disable=protected-access
        return copy

    def load(self, raise_errors=False, include_binary=True):
        """Load model components.

        Parameters
        ----------
        raise_errors : bool
            Error handling mode. If True, errors will be raised and stop loading.
            If False, errors will be printed but do not stop loading.
        include_binary : bool
            Read data from binary files in RESULTS folder. Default to True.

        Returns
        -------
        out : Field
            Field with loaded components.
        """
        name = os.path.basename(self._path)
        fmt = os.path.splitext(name)[1].strip('.')

        if fmt.upper() == 'HDF5':
            self._load_hdf5(raise_errors=raise_errors)
        elif fmt.upper() in ['DATA', 'DAT']:
            self._load_data(raise_errors=raise_errors, include_binary=include_binary)
        else:
            raise NotImplementedError('Format {} is not supported.'.format(fmt))

        self._collect_loaded_attrs()
        return self

    def _load_hdf5(self, raise_errors):
        """Load model in HDF5 format."""
        with h5py.File(self.path, 'r') as f:
            for k, v in f.attrs.items():
                if k == 'DATES':
                    self.meta['DATES'] = pd.to_datetime(v)
                else:
                    self.meta[k] = v
        for comp, config in self._config.items():
            getattr(self, comp).load(self.path,
                                     attrs=config['attrs'],
                                     raise_errors=raise_errors,
                                     logger=self._logger,
                                     **config['kwargs'])
        return self

    def _load_binary(self, components, raise_errors):
        """Load data from binary files in RESULTS folder."""
        path_to_results = os.path.join(os.path.dirname(self.path), 'RESULTS')
        if not os.path.exists(path_to_results):
            if raise_errors:
                raise ValueError("RESULTS folder was not found in model directory.")
            self._logger.warning("RESULTS folder was not found in model directory.")
            return
        for comp in components:
            if comp in self._config:
                getattr(self, comp).load(path_to_results,
                                         attrs=self._config[comp]['attrs'],
                                         basename=self.basename,
                                         logger=self._logger,
                                         **self._config[comp]['kwargs'])

    def _get_loaders(self, config):
        loaders = {}
        for k in META_KW:
            loaders[k] = partial(self._read_buffer, attr=k, logger=self._logger)

        loaders['COPY'] = partial(load_copy, self, logger=self._logger)
        loaders['MULTIPLY'] = partial(load_multiply, self, logger=self._logger)
        loaders['EQUALS'] = partial(load_equals, self, logger=self._logger)
        loaders['ADD'] = partial(load_add, self, logger=self._logger)
        for comp, conf in config.items():
            if conf['attrs'] is not None:
                attrs = list(set(conf['attrs']) - set(getattr(self, comp).state.binary_attributes))
            else:
                attrs = None
            kwargs = conf['kwargs']
            if comp in ['grid', 'rock', 'states', 'tables', 'faults']:
                assert attrs is not None
                for k in attrs:
                    loaders[k] = partial(getattr(self, comp).load, attr=k,
                                         logger=self._logger, **kwargs)
            if comp == 'wells':
                extented_list = []
                assert attrs is not None
                for k in attrs:
                    if k in ['PERF', 'EVENTS']:
                        extented_list.extend(['EFIL', 'EFILE', 'ETAB'])
                    elif k == 'HISTORY':
                        extented_list.extend(['HFIL', 'HFILE'])
                    elif k == 'WELLTRACK':
                        extented_list.extend(['TFIL', 'WELLTRACK'])
                    elif k == 'RESULTS':
                        continue
                    else:
                        extented_list.append(k)
                if kwargs.get('groups', True):
                    extented_list.extend(['GROU', 'GROUP', 'GRUPTREE'])

                for k in set(extented_list):
                    loaders[k] = partial(self.wells.load, attr=k, logger=self._logger,
                                         meta=self.meta, grid=self.grid, **kwargs)
            if comp == 'aquifers':
                for k in ['AQCT', 'AQCO', 'AQUANCON', 'AQUCT']:
                    loaders[k] = partial(self.aquifers.load, attr=k, logger=self._logger)
        return loaders

    def _load_results(self, raise_errors, include_binary):
        config = self._config
        if (('wells' in config) and ('attrs' in config['wells']) and
            ('RESULTS' in config['wells']['attrs'])):
            path_to_results = os.path.join(os.path.dirname(self.path), 'RESULTS')
            rsm = get_single_path(path_to_results, self.basename + '.RSM', self._logger)
            if rsm is None and not include_binary:
                if raise_errors:
                    raise ValueError("RSM file was not found in model directory.")
                self._logger.warning("RSM file was not found in model directory.")
            if rsm is not None and 'RESULTS' not in self.wells.state.binary_attributes:
                self.wells.load(rsm, logger=self._logger)
        return self

    def _check_vapoil(self):
        if 'VAPOIL' in self.meta['FLUIDS'] and 'tables' in self.components:
            # TODO should we make a kwarg for convertion key?
            self.tables.pvtg_to_pvdg(as_saturated=False)
            self.meta['FLUIDS'].remove('VAPOIL')
            self._logger.warning(
                """Vaporized oil option is not currently supported.
                PVTG table is converted into PVDG one."""
            )
        return self

    def _load_data(self, raise_errors=False, include_binary=True):
        """Load model in DATA format."""
        if include_binary:
            self._load_binary(components=('grid',),
                              raise_errors=raise_errors)
            if 'ACTNUM' in self.grid.state.binary_attributes:
                self._load_binary(components=('rock',), raise_errors=raise_errors)

        loaders = self._get_loaders(self._config)
        tnav_ascii_parser(self._path, loaders, self._logger, encoding=self._encoding,
                          raise_errors=raise_errors)

        self.grid = specify_grid(self.grid)

        if 'MINPV' in self.grid.attributes:
            if 'ACTNUM' in self.grid.state.binary_attributes:
                self._logger.info('ACTNUM is loaded from binary file: MINPV was not applied.')
            else:
                self.grid.apply_minpv()
                self._logger.info('MINPV {} is applied.'.format(self.grid.minpv[0]))

        if include_binary:
            self._load_binary(components=('states', 'wells'), raise_errors=raise_errors)

        self._load_results(raise_errors, include_binary)
        self._check_vapoil()

        if 'wells' in self.components:
            self.wells.add_welltrack()
            for well in self.wells:
                if 'COMPDAT' in well or 'COMPDATL' in well:
                    self.meta['MODEL_TYPE'] = 'ECL'
                    break
            else:
                self.meta['MODEL_TYPE'] = 'TN'
            self._logger.info('Model type is determined as {}.'.format(self.meta['MODEL_TYPE']))


        if self._config['grid']['kwargs'].get('apply_mapaxes', False):
            self.grid.map_grid()
            self._logger.info('Grid pillars `COORD` are mapped to new axis with respect to `MAPAXES`.')

        if 'states' in self.components:
            if not self.states.state.binary_attributes and self.states.attributes:
                self.states.dates = pd.to_datetime([self.meta['START']])
                self._logger.info('States dates are set to start date {}.'.format(self.meta['START']))

        return self

    def _read_buffer(self, buffer, attr, logger):
        """Load model meta attributes."""
        if attr in ['TITLE', 'START']:
            self.meta[attr] = next(buffer).split('/')[0].strip(' \t\n\'\""')
        elif attr == 'DATES':
            date = pd.to_datetime(next(buffer).split('/')[:1])
            self.meta['DATES'] = self.meta['DATES'].append(date)
        elif attr in ['ARRA', 'ARRAY']:
            dates = read_dates_from_buffer(buffer, attr, logger)
            self.meta['DATES'] = self.meta['DATES'].append(dates)
        elif attr in ['METRIC', 'FIELD']:
            self.meta['UNITS'] = attr
        elif attr in ['HUNI', 'HUNITS']:
            self._read_hunits(next(buffer))
        elif attr in ['OIL', 'GAS', 'WATER', 'DISGAS', 'VAPOIL']:
            self.meta['FLUIDS'].append(attr)
        elif attr in SUMMARY_KW:
            self.meta['SUMMARY'].append(attr)
        else:
            raise NotImplementedError("Keyword {} is not supported.".format(attr))
        return self

    def _read_hunits(self, line):
        """Parse HUNIts from line."""
        units = line.strip('/\t\n ').split()
        self.meta['HUNITS'] = []
        defaults = DEFAULT_HUNITS[self.meta['UNITS']]
        for k in units:
            if '*' in k:
                n = int(k[0])
                nread = len(self.meta['HUNITS'])
                self.meta['HUNITS'].extend(defaults[nread: nread + n])
            else:
                self.meta['HUNITS'].append(k)
        assert len(self.meta['HUNITS']) == len(defaults), 'Missmatch of HUNITS array length'
        return self

    def _collect_loaded_attrs(self):
        """Collect loaded attributes."""
        out = {}
        self._logger.info("===== Field summary =====")
        for comp in self.components:
            if comp == 'wells':
                attrs = []
                for node in PreOrderIter(self.wells.root):
                    attrs.extend(list(node.attributes))
                attrs = list(set(attrs))
            elif comp == 'faults':
                attrs = []
                for node in PreOrderIter(self.faults.root):
                    attrs.extend(list(node.attributes))
                attrs = list(set(attrs))
            elif comp == 'aquifers':
                attrs = []
                for _, aqf in self.aquifers.items():
                    attrs.extend(list(aqf.attributes))
                attrs = list(set(attrs))
            else:
                attrs = getattr(self, comp).attributes
            msg = "{} attributes: {}".format(comp.upper(), ', '.join(attrs))
            out[comp.upper()] = attrs
            self._logger.info(msg)
        self._logger.info("=========================")
        return out

    def dump(self, path=None, mode='a', data=True, results=True, title=None, **kwargs):
        """Dump model components.

        Parameters
        ----------
        path : str
            Common path for output files. If None path will be inherited from model path.
        mode : str
            Mode to open file. Affects only HDF5 dump.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'a'.
        data : bool
            Dump initial model data. No effect for HDF5 or VTU output. Default True.
        results : bool
            Dump calculated results. No effect for HDF5 or VTU output. Default True.
        title : str
            Model name. No effect for HDF5 or VTU output.
        kwargs : misc
            Any additional named arguments to ``dump``.

        Returns
        -------
        out : Field
            Field unchanged.
        """
        if title is None:
            title = self.meta.get('TITLE', 'Untitled')
        if path is None:
            dir_path = str(self._path.parent)
            return self._dump_binary_results(dir_path, mode, title=title)
        name = os.path.basename(path)
        fmt = os.path.splitext(name)[1].strip('.')
        if fmt.upper() == 'HDF5':
            return self._dump_hdf5(path, mode=mode, **kwargs)
        if fmt.upper() == 'VTU':
            dataset = self.get_vtk_dataset()
            writer = vtkXMLUnstructuredGridWriter()
            writer.SetFileName(path)
            writer.SetInputData(dataset)
            writer.Write()
            return self
        if fmt == '':
            dir_path = os.path.join(path, title)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            if data:
                self._dump_ascii(dir_path, title=title, **kwargs)
            if results and not self.wells.result_dates.empty:
                self._dump_binary_results(dir_path, mode, title=title)
        else:
            raise NotImplementedError('Format {} is not supported.'.format(fmt))
        return self

    def _dump_hdf5(self, path, mode='w', only_active=False, reduce_floats=True, **kwargs):
        """Dump model into HDF5 file.

        Parameters
        ----------
        path : str
            Path to output file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'w'.
        only_active : bool
            Keep state values in active cells only. Default to False.
        reduce_floats : bool
            if True, float precision will be reduced to np.float32 to save disk space.
        kwargs : misc
            Any additional named arguments to component's ``dump``.

        Returns
        -------
        out : Field
            Field unchanged.
        """
        float_precision = {
            'grid': np.float32 if reduce_floats else None,
            'rock': np.float32 if reduce_floats else None,
            'states': np.float32 if reduce_floats else None,
        }
        with h5py.File(path, mode) as f:
            for k, v in self.meta.items():
                if k == 'DATES':
                    f.attrs['DATES'] = pd.to_datetime(self.meta['DATES']).astype(np.int64)
                else:
                    f.attrs[k] = v
        for k, comp in self.items():
            if k == 'states':
                comp.dump(path=path, mode='a', float_dtype=float_precision.get(k, None),
                          actnum=self.grid.actnum if only_active else None, **kwargs)
            else:
                comp.dump(path=path, mode='a', float_dtype=float_precision.get(k, None), **kwargs)
        return self

    def _dump_ascii(self, dir_path, title, **kwargs):
        """Dump model's data files in tNav project.

        Parameters
        ----------
        dir_path : str
            Directory where model files will be placed.
        title : str
            Model title.
        kwargs : misc
            Any additional named arguments that will be substituted to model template file.

        Returns
        -------
        out : Field
            Field unchanged.
        """
        dir_inc = os.path.join(dir_path, 'INCLUDE')
        if not os.path.exists(dir_inc):
            os.mkdir(dir_inc)
        datafile = os.path.join(dir_path, title + '.data')

        model_type = self.meta.get('MODEL_TYPE', 'TN')
        if model_type == 'TN':
            template = Template(DEFAULT_TN_MODEL)
        elif model_type == 'ECL':
            template = Template(DEFAULT_ECL_MODEL)
        else:
            raise ValueError('Unknown model type {}'.format(model_type))

        fill_values = {'title': title, 'units': self.meta['UNITS']}
        fill_values['phases'] = '\n'.join(self.meta['FLUIDS'])

        if 'START' in self.meta:
            fill_values['start'] = self.meta['START']

        fill_values['dates'] = dates_to_str(self.result_dates)

        fill_values['dimens'] = ' '.join(self.grid.dimens.astype(str))
        fill_values['size'] = np.prod(self.grid.dimens)

        def compressed_str(arr):
            "Compressed array string."
            out = ''
            d = np.hstack([[0], np.where(np.diff(arr) != 0)[0]+1, [len(arr)]])
            for i in range(len(d)-1):
                val = arr[d[i]]
                count = d[i+1] - d[i]
                if count > 1:
                    out += '{}*{} '.format(count, val)
                else:
                    out += '{} '.format(val)
            return out + '/'

        if isinstance(self.grid, OrthogonalGrid):
            template = Template(template.safe_substitute(grid_specs=ORTHOGONAL_GRID))
            fill_values.update(dict(
                mapaxes=' '.join(self.grid.mapaxes.astype(str)),
                dx=compressed_str(self.grid.ravel('dx')),
                dy=compressed_str(self.grid.ravel('dy')),
                dz=compressed_str(self.grid.ravel('dz')),
                tops=compressed_str(self.grid.ravel('tops'))
            ))
        elif isinstance(self.grid, CornerPointGrid):
            template = Template(template.safe_substitute(grid_specs=CORNERPOINT_GRID))
            coord = os.path.join('INCLUDE', 'coord.inc')
            self.grid.dump(os.path.join(dir_path, coord), attrs='COORD', compressed=False)
            zcorn = os.path.join('INCLUDE', 'zcorn.inc')
            self.grid.dump(os.path.join(dir_path, zcorn), attrs='ZCORN')
            fill_values.update({'coord': coord, 'zcorn': zcorn})
        else:
            raise NotImplementedError("Dump for grid of type {} is not implemented."
                                      .format(self.grid.__class__.__name__))

        inc = os.path.join('INCLUDE', 'actnum.inc')
        self.grid.dump(os.path.join(dir_path, inc), attrs='ACTNUM', fmt='%i')
        fill_values['actnum'] = inc

        tmp = ''
        for attr_name, comp_name in SECTIONS_DICT['GRID']:
            if attr_name in getattr(self, comp_name).attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr_name.lower())
                inc = os.path.join('INCLUDE', attr_name.lower() + '.inc')
                getattr(self, comp_name).dump(os.path.join(dir_path, inc), attrs=attr_name, fmt='%.3f')
                fill_values[attr_name.lower()] = inc
        template = Template(template.safe_substitute(rock_grid=tmp))

        if 'faults' in self.components:
            attrs = ['faults', 'multflt']
            for attr in attrs:
                inc = os.path.join('INCLUDE', attr + '.inc')
                self.faults.dump(os.path.join(dir_path, inc), attr=attr)
                fill_values[attr] = inc

        tmp = ''
        if 'aquifers' in self.components:
            tmp += "INCLUDE\n'${} '/\n/\n\n".format('aquifers_file')
            inc = os.path.join('INCLUDE', 'aquifers.inc')
            self.aquifers.dump(os.path.join(dir_path, inc))
            fill_values['aquifers_file'] = inc
        template = Template(template.safe_substitute(aquifers=tmp))

        if model_type == 'TN':
            attrs = ['welltrack', 'perf', 'group', 'events']
        elif model_type == 'ECL':
            attrs = ['gruptree', 'schedule', 'welspecs']
        else:
            raise ValueError(f'Model type {model_type} is not supported.')

        for attr in attrs:
            inc = os.path.join('INCLUDE', attr + '.inc')
            self.wells.dump(os.path.join(dir_path, inc), attr=attr, grid=self.grid,
                            dates=self.result_dates, start_date=self.start)
            fill_values[attr] = inc

        tmp = ''
        if 'states' in self.components:
            for attr in self.states.attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr.lower())
                inc = os.path.join('INCLUDE', attr.lower() + '.inc')
                self.states.dump(os.path.join(dir_path, inc), attrs=attr, fmt='%.3f')
                fill_values[attr.lower()] = inc
        template = Template(template.safe_substitute(states=tmp))

        tmp = ''
        if 'tables' in self.components:
            for attr in self.tables.attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr.lower())
                inc = os.path.join('INCLUDE', attr.lower() + '.inc')
                self.tables.dump(os.path.join(dir_path, inc), attrs=attr)
                fill_values[attr.lower()] = inc
        template = Template(template.safe_substitute(tables=tmp))
        tmp = ''
        for attr_name, comp_name in SECTIONS_DICT['PROPS']:
            if attr_name in getattr(self, comp_name).attributes:
                tmp += "INCLUDE\n'${}'\n\n".format(attr_name.lower())
                inc = os.path.join('INCLUDE', attr_name.lower() + '.inc')
                getattr(self, comp_name).dump(os.path.join(dir_path, inc), attrs=attr_name, fmt='%.3f')
                fill_values[attr_name.lower()] = inc
        template = Template(template.safe_substitute(rock_props=tmp))
        template = Template(template.safe_substitute(fill_values))
        template = Template(template.safe_substitute(kwargs))

        out = template.safe_substitute()
        missing = Template.pattern.findall(out)
        if missing:
            self._logger.warning('Dump missed values for %s.', ', '.join([i[1] for i in missing]))

        with open(datafile, 'w') as f:
            f.writelines(out)
        return self

    def _dump_binary_results(self, dir_path, mode, title):
        """Dump model's binary result files.

        Parameters
        ----------
        dir_path : str
            Path to location where RESULTS directory will be created.
        title : str
            Model title.

        Returns
        -------
        out : Field
            Field unchanged.
        """
        dir_res = os.path.join(dir_path, 'RESULTS')
        if not os.path.exists(dir_res):
            os.mkdir(dir_res)

        grid_dim = self.grid.dimens
        time_size = self.states.n_timesteps

        dir_name = os.path.join(dir_res, title)

        units_type = {
            'METRIC': 1,
            'FIELD': 2,
            'LAB': 3,
            'PVT-M': 4,
        }[self.meta['UNITS']]

        grid_type = {
            'CornerPointGrid': 0,
            'OrthogonalGrid': 3,
        }[self.grid.class_name]

        grid_format = {
            'CornerPointGrid': 1,
            'OrthogonalGrid': 2,
        }.get(self.grid.class_name, 0)# 0 - Unknown; 1 - Corner point; 2 - Block centered

        i_phase = 0
        for elem in self.meta['FLUIDS']:
            i_phase += {'OIL': 1, 'WATER': 2, 'GAS': 4}.get(elem, 0)

        egrid.save_egrid(self.grid.as_corner_point, dir_name, grid_dim, grid_format, mode)

        init.save_init(self.rock, dir_name, grid_dim, self.grid.actnum.sum(),
                       units_type, grid_type, self.start, i_phase, mode)

        is_unified = True

        restart.save_restart(is_unified,
                             dir_name,
                             self.states.strip_na(),
                             self.states.attributes,
                             self.states.dates,
                             grid_dim,
                             time_size,
                             mode,
                             self._logger)

        rates = {}
        for well in self.wells.main_branches:
            curr_rates = self.wells[well].total_rates
            well_data = {}
            for k in ['WWPR', 'WOPR', 'WGPR']:
                if k in curr_rates:
                    well_data[k.lower()] = curr_rates[k].values.astype('float64')
            if well_data:
                rates[well] = well_data

        summary.save_summary(is_unified, dir_name, rates, self.result_dates,
                             grid_dim, mode, self._logger)
        return self

    def calculate_rates(self, wellnames=None, cf_aggregation='sum', multiprocessing=True, verbose=True):
        """Calculate oil/water/gas rates for each well segment.
        NOTE: Rate calculation is supported only for three phase fluid with dissolved gas (
            OIL, WATER, GAS, DISGAS). Eclipse style control is not supported.

        Parameters
        ----------
        wellnames : list of str
            Wellnames for rates calculation. If None, all wells are included. Default None.
        cf_aggregation: str, 'sum' or 'eucl'
            The way of aggregating cf projection ('sum' - sum, 'eucl' - Euclid norm).
        multiprocessing : bool
            Use multiprocessing for rates calculation. Default True.
        verbose : bool
            Print a number of currently processed wells (if multiprocessing=False). Default True.

        Returns
        -------
        model : Field
            Reservoir model with computed rates.
        """
        timesteps = self.result_dates
        if wellnames is None:
            wellnames = self.wells.names
        if multiprocessing:
            calc_rates_multiprocess(self, timesteps, wellnames, cf_aggregation)
        else:
            calc_rates(self, timesteps, wellnames, cf_aggregation, verbose)
        return self

    def history_to_results(self):
        """Convert history to results."""
        for node in self.wells:
            if node.is_main_branch and not hasattr(node, 'history'):
                self.wells.drop(node.name)

        for node in self.wells:
            if hasattr(node, 'results'):
                delattr(node, 'results')

        rename = {'QOIL': 'WOPR', 'QGAS': 'WGPR', 'QWAT': 'WWPR', 'QWIN': 'WWIR', 'BHP': 'WBHP'}
        for node in self.wells:
            if node.is_main_branch:
                new_results = node.history.rename(columns=rename)[['DATE'] + list(rename.values())]
                node.results = new_results.drop_duplicates(subset='DATE')
                node.results = node.results.reset_index(drop=True)
        return self

    # pylint: disable=protected-access
    def get_vtk_dataset(self):
        """Create vtk dataset with data from `rock` and `states` components.
        Grid is represented in unstructured form.

        Returns
        -------
        vtk.vtkUnstructuredGrid
            vtk dataset with states and rock data.

        """
        grid = self.grid.to_corner_point()
        vtk_grid_old = grid._vtk_grid
        grid.create_vtk_grid()
        dataset = grid._vtk_grid

        for comp_name in ('rock', 'states'):
            comp = getattr(self, comp_name)
            for attr in comp.attributes:
                val = getattr(comp, attr)
                if val.ndim == 3:
                    array = numpy_to_vtk(val[grid.actnum])
                elif val.ndim == 4:
                    array = numpy_to_vtk(val[:, grid.actnum].T)
                else:
                    raise ValueError('Attribute {attr} in component {comp_name}' +
                                     'should be 3 or 4 dimensional array to be dumped.')
                array.SetName('_'.join((comp_name.upper(), attr)))
                dataset.GetCellData().AddArray(array)
        ind_i, ind_j, ind_k = np.indices(grid.dimens)
        for name, val in zip(('I', 'J', 'K'), (ind_i, ind_j, ind_k)):
            array = numpy_to_vtk(val[grid.actnum])
            array.SetName(name)
            dataset.GetCellData().AddArray(array)
        grid._vtk_grid = vtk_grid_old
        return dataset

    # pylint: disable=protected-access
    def _create_pyvista_grid(self):
        """Creates pyvista grid object with attributes."""
        self._pyvista_grid = pv.UnstructuredGrid(self.grid._vtk_grid)

        attributes = {}
        active_cells = self.grid.actnum if 'ACTNUM' in self.grid else np.full(self.grid.dimens, True)

        def make_data(data):
            if self.grid._vtk_grid_params['use_only_active']:
                return data[active_cells].astype(float)
            new_data = data.copy()
            new_data[~active_cells] = np.nan
            return new_data.ravel().astype(float)

        attributes.update({'ACTNUM': make_data(active_cells)})

        if 'rock' in self.components:
            attributes.update({attr: make_data(data) for attr, data in self.rock.items()})

        if 'states' in self.components:
            for attr, sequence in self.states.items():
                attributes.update({'%s_%d' % (attr, i): make_data(snapshot)
                                   for i, snapshot in enumerate(sequence)})

        self._pyvista_grid.cell_data.update(attributes)

    def _add_welltracks(self, plotter):
        """Adds all welltracks to the plot."""
        well_tracks = {node.name: self.wells[node.name].welltrack[:, :3].copy()
                       for node in self.wells if 'WELLTRACK' in node.attributes}

        labeled_points = {}
        dz = self._pyvista_grid.bounds[5] - self._pyvista_grid.bounds[4]
        z_min = self._pyvista_grid.bounds[4] - 0.05 * dz

        vertices = []
        faces = []
        size = 0
        for well_name, value in well_tracks.items():
            wtrack_idx, first_intersection = self.wells[well_name]._first_entering_point # pylint: disable=protected-access
            if first_intersection is not None:
                value = np.concatenate([np.array([[first_intersection[0], first_intersection[1], z_min]]),
                                        np.asarray(first_intersection).reshape(1, -1),
                                        value[wtrack_idx + 1:]])
            else:
                value = np.concatenate([np.array([[value[0, 0], value[0, 1], z_min]]), value])

            vertices.append(value)
            ids = np.arange(size, size+len(value))
            faces.append(np.stack([0*ids[:-1]+2, ids[:-1], ids[1:]]).T)
            size += len(value)
            labeled_points[well_name] = value[0]

        mesh = pv.PolyData(np.vstack(vertices), lines=np.vstack(faces))
        plotter.add_mesh(mesh, name='wells', color='k', line_width=2)

        return labeled_points

    def _add_faults(self, plotter, use_only_active=True, color='red'):
        """Adds all faults to the plot."""
        faces = []
        vertices = []
        labeled_points = {}
        size = 0
        for segment in self.faults:
            blocks = segment.blocks
            xyz = segment.faces_verts
            if use_only_active:
                active = self.grid.actnum[blocks[:, 0], blocks[:, 1], blocks[:, 2]]
                xyz = xyz[active]
            if len(xyz) == 0:
                continue
            vertices.append(xyz.reshape(-1, 3))
            ids = np.arange(size, size+4*len(xyz))
            faces1 = np.stack([0*ids[::4]+3, ids[::4], ids[1::4], ids[3::4]]).T
            faces2 = np.stack([0*ids[::4]+3, ids[::4], ids[2::4], ids[3::4]]).T
            size += 4*len(xyz)
            faces.extend([faces1, faces2])
            labeled_points[segment.name] = xyz[0, 0]

        if faces:
            mesh = pv.PolyData(np.vstack(vertices), np.vstack(faces))
            plotter.add_mesh(mesh, name='faults', color=color)

        return labeled_points

    def show(self, attr=None, thresholding=False, slicing=False, timestamp=None,
             use_only_active=True, scaling=True, cmap=None, notebook=False,
             theme='default', show_edges=True, faults_color='red', show_labels=True):
        """Field visualization.

        Parameters
        ----------
        attr: str or None
            Attribute of the grid to show. If None, ACTNUM will be shown.
        thresholding: bool
            Show slider for thresholding. Cells with attribute value less than
            threshold will not be shown. Default False.
        slicing: bool
            Show by slices. Default False.
        timestamp: int or None
            The timestamp to show. Meaningful only for sequential attributes (States).
            Has no effect given non-sequential attributes.
        use_only_active: bool
            Corner point grid creation using only active cells. Default to True.
        scaling: bool, list or tuple
            The ratio of the axes in case of iterable, if True then it's (1, 1, 1),
            if False then no scaling is applied. Default True.
        cmap: object
            Matplotlib, Colorcet, cmocean, or custom colormap
        notebook: bool
            When True, the resulting plot is placed inline a jupyter notebook.
            Assumes a jupyter console is active. Automatically enables off_screen.
        theme: str
            PyVista theme, e.g. 'default', 'dark', 'document', 'ParaView'.
            See https://docs.pyvista.org/examples/02-plot/themes.html for more options.
        show_edges: bool
            Shows the edges of a mesh. Default True.
        faults_color: str
            Corol to show faults. Default 'red'.
        show_labels: bool
            Show x, y, z axis labels. Default True.
        """
        attribute = 'ACTNUM' if attr is None else attr.upper()
        grid_params = {'use_only_active': use_only_active, 'scaling': scaling}

        plot_params = {'show_edges': show_edges, 'cmap': cmap}

        if 'wells' in self.components:
            self.wells._get_first_entering_point()

        if 'faults' in self.components:
            self.faults.get_blocks()

        old_vtk_grid_params = self.grid._vtk_grid_params

        if self.grid._vtk_grid is None or old_vtk_grid_params != grid_params:
            self.grid.create_vtk_grid(**grid_params)

        if self._pyvista_grid is None or old_vtk_grid_params != grid_params:
            self._create_pyvista_grid()

        grid = self._pyvista_grid

        sequential = ('states' in self.components) and (attribute in self.states)

        pv.set_plot_theme(theme)
        plotter = pv.Plotter(notebook=notebook, title='Field')
        plotter.set_viewup([0, 0, -1])
        plotter.set_position([1, 1, -0.3])

        threshold_widget = thresholding
        timestamp_widget = sequential and timestamp is None
        slice_xyz_widget = slicing

        if not thresholding:
            if sequential:
                threshold = np.min([np.nanmin(grid.cell_data[f"{attribute}_{i}"]) for i in
                                    range(self.states.n_timesteps)])
            else:
                threshold = np.nanmin(grid.cell_data[attribute])
        else:
            threshold = 0

        scaling = np.asarray(scaling).ravel()
        if len(scaling) == 1:
            if scaling[0]:
                scales = np.diff(self.grid.bounding_box, axis=0).ravel()
                scaling = scales.max() / scales #scale to unit cube
            else:
                scaling = np.array([1, 1, 1]) #no scaling

        widget_values = {
            'plotter': plotter,
            'grid': grid,
            'attribute': attribute,
            'opacity': 0.5,
            'threshold': threshold,
            'slice_xyz': np.array(grid.bounds).reshape(3, 2).mean(axis=1) if slicing else None,
            'timestamp': None if not sequential else 0 if timestamp is None else timestamp,
            'plot_params': plot_params,
            'scaling': scaling
        }

        def _create_mesh_wrapper(**kwargs):
            widget_values.update(kwargs)
            return create_mesh(**kwargs)

        plotter = _create_mesh_wrapper(**widget_values)

        slider_positions = [
            {'pointa': (0.03, 0.90), 'pointb': (0.30, 0.90)},
            {'pointa': (0.36, 0.90), 'pointb': (0.63, 0.90)},
            {'pointa': (0.69, 0.90), 'pointb': (0.97, 0.90)}
        ]

        slicing_slider_positions = [
            {'pointa': (0.03, 0.76), 'pointb': (0.30, 0.76)},
            {'pointa': (0.03, 0.62), 'pointb': (0.30, 0.62)},
            {'pointa': (0.03, 0.48), 'pointb': (0.30, 0.48)}
        ]

        def ch_opacity(x):
            return _create_mesh_wrapper(
                opacity=x, **{k: v for k, v in widget_values.items() if k != 'opacity'}
            )
        slider_pos = slider_positions.pop(0)
        slider_range = [0., 1.]
        plotter.add_slider_widget(ch_opacity, rng=slider_range, title='Opacity', **slider_pos)

        if threshold_widget:
            def ch_threshold(x):
                return _create_mesh_wrapper(
                    threshold=x, **{k: v for k, v in widget_values.items() if k != 'threshold'}
                )
            slider_pos = slider_positions.pop(0)
            if sequential:
                trange = np.arange(self.states.n_timesteps) if timestamp is None else [timestamp]
                min_slider = min((np.nanmin(grid.cell_data[f"{attribute}_{i}"]) for i in trange))
                max_slider = max((np.nanmax(grid.cell_data[f"{attribute}_{i}"]) for i in trange))
                slider_range = [min_slider, max_slider]
            else:
                slider_range = [np.nanmin(grid.cell_data[attribute]),
                                np.nanmax(grid.cell_data[attribute])]
            plotter.add_slider_widget(ch_threshold, rng=slider_range, title='Threshold', **slider_pos)

        if timestamp_widget:
            def ch_timestamp(x):
                return _create_mesh_wrapper(
                    timestamp=int(np.rint(x)), **{k: v for k, v in widget_values.items() if k != 'timestamp'}
                )
            slider_pos = slider_positions.pop(0)
            slider_range = [0, self.states.n_timesteps - 1]
            plotter.add_slider_widget(ch_timestamp, rng=slider_range, value=0,
                                      title='Timestamp', **slider_pos)

        if slice_xyz_widget:
            def ch_slice_x(x):
                new_slice_xyz = list(widget_values['slice_xyz'])
                new_slice_xyz[0] = x
                return _create_mesh_wrapper(
                    slice_xyz=tuple(new_slice_xyz), **{k: v for k, v in
                                                       widget_values.items() if k != 'slice_xyz'}
                )

            def ch_slice_y(y):
                new_slice_xyz = list(widget_values['slice_xyz'])
                new_slice_xyz[1] = y
                return _create_mesh_wrapper(
                    slice_xyz=tuple(new_slice_xyz), **{k: v for k, v in
                                                       widget_values.items() if k != 'slice_xyz'}
                )

            def ch_slice_z(z):
                new_slice_xyz = list(widget_values['slice_xyz'])
                new_slice_xyz[2] = z
                return _create_mesh_wrapper(
                    slice_xyz=tuple(new_slice_xyz), **{k: v for k, v in
                                                       widget_values.items() if k != 'slice_xyz'}
                )
            x_pos, y_pos, z_pos = slicing_slider_positions
            x_min, x_max, y_min, y_max, z_min, z_max = grid.bounds
            plotter.add_slider_widget(ch_slice_x, rng=[x_min, x_max], title='X', **x_pos)
            plotter.add_slider_widget(ch_slice_y, rng=[y_min, y_max], title='Y', **y_pos)
            plotter.add_slider_widget(ch_slice_z, rng=[z_min, z_max], title='Z', **z_pos)

        def show_wells(value=True):
            if value and ('wells' in self.components):
                labeled_points = self._add_welltracks(plotter)
                if labeled_points:
                    (labels, points) = zip(*labeled_points.items())
                    points = np.array(points)*scaling
                    plotter.add_point_labels(points, labels,
                        font_size=20,
                        show_points=False,
                        name='well_names')
            else:
                plotter.remove_actor('well_names')
                plotter.remove_actor('wells')
        show_wells()

        if not notebook:
            plotter.add_checkbox_button_widget(show_wells, value=True)
            plotter.add_text("      Wells", position=(10.0, 10.0), font_size=16)

        def show_faults(value=True):
            if value and ('faults' in self.components):
                labeled_points = self._add_faults(plotter,
                                                  use_only_active=use_only_active,
                                                  color=faults_color)
                if labeled_points:
                    (labels, points) = zip(*labeled_points.items())
                    points = np.array(points)*scaling
                    plotter.add_point_labels(points, labels,
                        font_size=20,
                        show_points=False,
                        name='fault_names')
            else:
                plotter.remove_actor('fault_names')
                plotter.remove_actor('faults')
        show_faults()

        if not notebook:
            plotter.add_checkbox_button_widget(show_faults, value=True, position=(10.0, 70.0))
            plotter.add_text("      Faults", position=(10.0, 70.0), font_size=16)

        plotter.show_grid(show_xlabels=show_labels, show_ylabels=show_labels, show_zlabels=show_labels)
        plotter.show()


def create_mesh(plotter, grid, attribute, opacity, threshold, slice_xyz, timestamp, plot_params, scaling):
    """Create mesh for pyvista visualisation."""
    plotter.remove_actor('cells')
    try:
        plotter.remove_scalar_bar()
    except IndexError:
        pass

    if timestamp is None:
        grid.set_active_scalars(attribute)
    else:
        grid.set_active_scalars('%s_%d' % (attribute, timestamp))

    if threshold is not None:
        grid = grid.threshold(threshold, continuous=True)

    if slice_xyz is not None:
        grid = grid.slice_orthogonal(x=slice_xyz[0], y=slice_xyz[1], z=slice_xyz[2])

    plot_params['scalar_bar_args'] = dict(title='', label_font_size=12, width=0.5, position_y=0.03, position_x=0.45)
    plotter.add_mesh(grid, name='cells', opacity=opacity, **plot_params)

    if timestamp is None:
        plotter.add_text(attribute, position='upper_edge', name='title', font_size=14)
    else:
        plotter.add_text('%s, t=%s' % (attribute, timestamp), position='upper_edge',
                         name='title', font_size=14)

    plotter.set_scale(*scaling)
    return plotter
