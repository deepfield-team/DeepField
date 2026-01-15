# pylint: disable=too-many-lines
"""Field class."""
import logging
import os
import pathlib
import sys
import weakref
from copy import deepcopy
from functools import partial
from string import Template
from anytree import PreOrderIter

import h5py
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk # pylint: disable=no-name-in-module, import-error

from .arithmetics import load_add, load_copy, load_equals, load_multiply
from .faults import Faults
from .aquifer import Aquifers
from .configs import default_config
from .dump_ecl_utils import egrid, init, restart, summary
from .grids import CornerPointGrid, Grid, OrthogonalGrid, specify_grid
from .parse_utils import (dates_to_str, preprocess_path,
                          read_dates_from_buffer, tnav_ascii_parser)
from .rock import Rock
from .states import States
from .tables import Tables
from .template_models import (CORNERPOINT_GRID, DEFAULT_ECL_MODEL,
                              DEFAULT_TN_MODEL, ORTHOGONAL_GRID)
from .utils import get_single_path
from .wells import Wells
import resdp
import resdp.binary

ACTOR = None

COMPONENTS_DICT = {
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
    def __init__(self, path: pathlib.Path | None=None, logfile=None, encoding='auto', loglevel='INFO'):
        self._path: pathlib.Path | None = preprocess_path(path) if path is not None else None
        self._encoding = encoding
        self._components = {}
        self._meta = {'UNITS': 'METRIC',
                      'START': pd.to_datetime(''),
                      'DATES': pd.to_datetime([]),
                      'FLUIDS': [],
                      'SUMMARY': [],
                      'MODEL_TYPE': '',
                      'HUNITS': DEFAULT_HUNITS['METRIC']}
        self._state = FieldState(self)
        self._data = resdp.DataType

        logging.shutdown()
        handlers = [logging.StreamHandler(sys.stdout)]
        if logfile is not None:
            handlers.append(logging.FileHandler(logfile, mode='w'))
        logging.basicConfig(handlers=handlers)
        self._logger = logging.getLogger('Field')
        self._logger.setLevel(getattr(logging, loglevel))

        if self._path is not None:
            self._init_components()

        self._pyvista_grid = None

    def _init_components(self):
        """Initialize components."""
        fmt = self._path.suffix.strip('.').upper()
        for k, (comp_name, comp_class) in COMPONENTS_DICT.items():
            setattr(self, comp_name, comp_class(field=self))

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

    @property
    def result_dates(self):
        """Result dates, actual if present, target otherwise."""
        return self.wells.result_dates

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
        if self._path is None:
            raise ValueError('Path to model is not defined.')
        name = os.path.basename(self._path)
        fmt = os.path.splitext(name)[1].strip('.')

        if fmt.upper() == 'HDF5':
            raise NotImplementedError('HDF5 format is not currently supported.')
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
        self.grid.create_vtk_grid()
        return self

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

        if self._path is None:
            raise ValueError()

        data = resdp.load(pathlib.Path(self.path))
        self._data = data
        if include_binary:
            self._binary_data = resdp.binary.load(pathlib.Path(self.path))
        else:
            self._binary_data = None

        for comp in self._components:
            getattr(self, comp).load(self._data, self._binary_data, self._logger)

        self.grid = specify_grid(self.grid)
        self.grid.create_vtk_grid()

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
        raise NotImplementedError('Dump is not implemented.')
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
        raise NotImplementedError('Dump to HDF5 is not implemented.')
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
        raise ValueError()
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
        dataset = vtk.vtkUnstructuredGrid()
        dataset.DeepCopy(self.grid.vtk_grid)
        actnum = self.grid.actnum.ravel(order='F')

        for comp_name in ('rock', 'states'):
            comp = getattr(self, comp_name)
            for attr in comp.attributes:
                val = getattr(comp, attr)
                if val.ndim == 3:
                    array = numpy_to_vtk(val.ravel(order='F')[actnum].astype('float32'))
                elif val.ndim == 4:
                    array = numpy_to_vtk(np.array([x.ravel(order='F')[actnum].astype('float32') for x in val]).T)
                else:
                    raise ValueError('Attribute {attr} in component {comp_name}' +
                                     'should be 3 or 4 dimensional array to be dumped.')
                array.SetName('_'.join((comp_name.upper(), attr)))
                dataset.GetCellData().AddArray(array)
        return dataset

    def _add_welltracks(self, plotter):
        """Adds all welltracks to the plot."""

        dz = self._pyvista_grid.bounds[5] - self._pyvista_grid.bounds[4]
        z_min = self._pyvista_grid.bounds[4] - 0.05 * dz

        vertices = []
        faces = []

        vertices_connectors = []
        labeled_points = {}

        size = 0
        for well in self.wells:
            if 'WELLTRACK' not in well:
                continue

            first_point = well.welltrack[0, :3].copy()
            first_point[-1] = z_min

            vertices.append(well.welltrack[:, :3])
            ids = np.arange(size, size+len(well.welltrack))
            faces.append(np.stack([0*ids[:-1]+2, ids[:-1], ids[1:]]).T)
            size += len(well.welltrack)

            vertices_connectors.extend([first_point, well.welltrack[0, :3]])
            labeled_points[well.name] = first_point

        vertices_connectors = np.array(vertices_connectors)
        count = len(vertices_connectors)
        faces_connectors = np.stack([np.full(count//2, 2),
                                     np.arange(0, count, 2),
                                     np.arange(1, count, 2)]).T

        if vertices:
            mesh = pv.PolyData(np.vstack(vertices), lines=np.vstack(faces))
            plotter.add_mesh(mesh, name='wells', color='b', line_width=3)

            mesh = pv.PolyData(vertices_connectors, lines=faces_connectors)
            plotter.add_mesh(mesh, name='well_connectors', color='k', line_width=2)

        return labeled_points

    def _add_faults(self, plotter, use_only_active=True, color='red'):
        """Adds all faults to the plot."""
        faces = []
        vertices = []
        labeled_points = {}
        size = 0
        for fault in self.faults:
            blocks = fault.blocks
            xyz = fault.faces_verts
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
            labeled_points[fault.name] = xyz[0, 0]

        if faces:
            mesh = pv.PolyData(np.vstack(vertices), np.vstack(faces))
            plotter.add_mesh(mesh, name='faults', color=color)

        return labeled_points

    def show(self, attr=None, thresholding=False, slicing=False, timestamp=None,
             scaling=True, cmap=None, notebook=False,
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
        if self._pyvista_grid is None:
            self._pyvista_grid = pv.UnstructuredGrid(self.grid.vtk_grid)

        if attr is not None:
            attr = attr.upper()
            sequential = ('states' in self.components) and (attr in self.states)
        else:
            sequential = False

        pv.set_plot_theme(theme)
        plotter = pv.Plotter(notebook=notebook, title='Field')
        plotter.set_viewup([0, 0, -1])
        plotter.set_position([1, 1, -0.3])

        threshold_widget = thresholding
        timestamp_widget = sequential and timestamp is None
        slice_xyz_widget = slicing

        scaling = np.asarray(scaling).ravel()
        bbox = self.grid.bounding_box
        if len(scaling) == 1:
            if scaling[0]:
                scales = bbox[3:] - bbox[:3]
                scaling = scales.max() / scales #scale to unit cube
            else:
                scaling = np.array([1, 1, 1]) #no scaling

        widget_values = {
            'plotter': plotter,
            'attribute': attr,
            'opacity': 0.5,
            'threshold': None,
            'slice_xyz': (bbox[3:] + bbox[:3])//2 if slicing else None,
            'timestamp': None if not sequential else 0 if timestamp is None else timestamp,
            'plot_params': {'show_edges': show_edges, 'cmap': cmap},
            'scaling': scaling
        }

        plotter = self._create_mesh(**widget_values)

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
            widget_values['opacity'] = x
            return self._create_mesh(**widget_values)

        slider_pos = slider_positions.pop(0)
        slider_range = [0., 1.]
        plotter.add_slider_widget(ch_opacity, rng=slider_range, title='Opacity', **slider_pos)

        if threshold_widget:
            def ch_threshold(x):
                widget_values['threshold'] = x
                return self._create_mesh(**widget_values)
            slider_pos = slider_positions.pop(0)
            comp = self.states if sequential else self.rock
            if attr is not None:
                slider_range = [np.nanmin(comp[attr]), np.nanmax(comp[attr])]
            else:
                slider_range = [0, 0]
            plotter.add_slider_widget(ch_threshold, rng=slider_range, title='Threshold', **slider_pos)

        if timestamp_widget:
            def ch_timestamp(x):
                widget_values['timestamp'] = int(np.rint(x))
                return self._create_mesh(**widget_values)
            slider_pos = slider_positions.pop(0)
            slider_range = [0, self.states.n_timesteps - 1]
            plotter.add_slider_widget(ch_timestamp, rng=slider_range, value=0,
                                      title='Timestamp', **slider_pos)

        if slice_xyz_widget:
            def ch_slice_x(x):
                widget_values['slice_xyz'][0] = x #pylint: disable=unsupported-assignment-operation
                return self._create_mesh(**widget_values)

            def ch_slice_y(y):
                widget_values['slice_xyz'][1] = y #pylint: disable=unsupported-assignment-operation
                return self._create_mesh(**widget_values)

            def ch_slice_z(z):
                widget_values['slice_xyz'][2] = z #pylint: disable=unsupported-assignment-operation
                return self._create_mesh(**widget_values)

            x_pos, y_pos, z_pos = slicing_slider_positions
            x_min, y_min, z_min, x_max, y_max, z_max = bbox
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
                plotter.remove_actor('well_connectors')
        show_wells()

        if not notebook:
            plotter.add_checkbox_button_widget(show_wells, value=True)
            plotter.add_text("      Wells", position=(10.0, 10.0), font_size=16)

        if 'faults' in self.components:
            self.faults.get_blocks()

        def show_faults(value=True):
            if value and ('faults' in self.components):
                labeled_points = self._add_faults(plotter,
                                                  use_only_active=True,
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

    def _create_mesh(self, plotter, attribute, opacity, threshold, slice_xyz,
                     timestamp, plot_params, scaling):
        """Create mesh for pyvista visualisation."""
        grid = self._pyvista_grid

        plotter.remove_actor('cells')
        try:
            plotter.remove_scalar_bar()
        except (IndexError, StopIteration):
            pass

        name = attribute if timestamp is None else '%s_%d' % (attribute, timestamp)
        if attribute is not None and name not in grid.cell_data:
            actnum = self.grid.actnum.ravel()
            data = self.rock[attribute] if timestamp is None else self.states[attribute][timestamp]
            data = data.ravel()[actnum]
            grid.cell_data[name] = data
        grid.set_active_scalars(name)

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
