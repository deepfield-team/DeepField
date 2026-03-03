# pylint: disable=too-many-lines
"""Field class."""
import logging
import os
import pathlib
import sys

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk # pylint: disable=no-name-in-module, import-error

import resdp
import resdp.binary

from .grids import Grid, specify_grid
from .rock import Rock
from .states import States
from .tables import Tables
from .wells import Wells
from .faults import Faults


ACTOR = None

COMPONENTS = [Grid, Rock, States, Wells, Tables, Faults]
COMPONENT_NAMES = [x.__name__.lower() for x in COMPONENTS]

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


class Field:
    """Reservoir model.

    Contains reservoir model data and data processing tools.

    Parameters
    ----------
    path : str, optional
        Path to source model files.
    logfile : str, optional
        Path to log file.
    loglevel : str, optional
        Log level to be printed while loading. Default to 'INFO'.
    """
    def __init__(self, path: pathlib.Path | None=None, logfile=None, loglevel='INFO'):
        self.path: pathlib.Path | None = pathlib.Path(path) if path is not None else None
        self._components = {}

        self._data = resdp.DataType
        self._binary_data = resdp.DataType | None

        logging.shutdown()
        handlers = [logging.StreamHandler(sys.stdout)]
        if logfile is not None:
            handlers.append(logging.FileHandler(logfile, mode='w'))
        logging.basicConfig(handlers=handlers)
        self._logger = logging.getLogger('Field')
        self._logger.setLevel(getattr(logging, loglevel))

        for comp in COMPONENTS:
            self._components[comp.__name__.lower()] = comp(field=self)

        self._pyvista_grid = None

    def __getattr__(self, attr):
        try:
            return self._components[attr]
        except KeyError:
            raise AttributeError(attr) #pylint: disable=raise-missing-from

    def __setattr__(self, attr, value):
        if attr in COMPONENT_NAMES:
            self._components[attr] = value
        else:
            super().__setattr__(attr, value)
        return self

    @property
    def name(self):
        """Model filename without extention."""
        fname = os.path.basename(self.path)
        return os.path.splitext(fname)[0]

    @property
    def components(self):
        """Model components."""
        return tuple(self._components.keys())

    @property
    def logger(self):
        "Logger."
        return self._logger

    def items(self):
        """Returns pairs of components's names and instance."""
        return self._components.items()

    def load(self, include_binary=True):
        """Load reservoir model data.

        Parameters
        ----------
        include_binary : bool
            Read data from binary files in RESULTS folder. Default to True.

        Returns
        -------
        out : Field
            Field with loaded components.
        """
        if self.path is None:
            raise ValueError('Path to the reservoir model is not specified.')

        name = os.path.basename(self.path)
        fmt = os.path.splitext(name)[1].strip('.')

        if fmt.upper() in ['DATA', 'DAT']:
            self._load_data(include_binary=include_binary)
        else:
            raise NotImplementedError('Format {} is not supported.'.format(fmt))

        self.grid = specify_grid(self.grid) #pylint: disable=attribute-defined-outside-init
        self.grid.create_vtk_grid()

        self.wells.build_tree()
        self.wells.fill_nan_coordinates()
        self.wells.add_welltrack(overwrite=False)

        self.faults.build_tree()

        self._collect_loaded_attrs()

        return self

    def _load_data(self, include_binary=True):
        """Load model in DATA format."""
        self._data = resdp.load(self.path)

        self._binary_data = resdp.binary.load(self.path) if include_binary else None

        for _, comp in self.items():
            comp.load(self._data, self._binary_data, self._logger)

        return self

    def _collect_loaded_attrs(self):
        """Collect loaded attributes."""
        out = {}
        self._logger.info("===== Field summary =====")
        for k, comp in self.items():
            attrs = comp.attributes
            msg = "{} attributes: {}".format(k.upper(), ', '.join(attrs))
            out[k.upper()] = attrs
            self._logger.info(msg)
        self._logger.info("=========================")
        return out

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
                    raise ValueError('Attribute {attr} in component {comp_name} should be 3D or 4D array.')
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

            welltrack = well.welltrack[['X', 'Y', 'Z']].values
            first_point = welltrack[0, :3].copy()
            first_point[-1] = z_min

            vertices.append(welltrack[:, :3])
            ids = np.arange(size, size+len(welltrack))
            faces.append(np.stack([0*ids[:-1]+2, ids[:-1], ids[1:]]).T)
            size += len(welltrack)

            vertices_connectors.extend([first_point, welltrack[0, :3]])
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

    def show(self, attr=None, thresholding=False, slicing=False, timestamp=None, opacity=0.5,
             scaling=True, cmap=None, notebook=False, backend=None,
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
        opacity : float
            Opacity value between 0 and 1. Default 0.5.
        scaling: bool, list or tuple
            The ratio of the axes in case of iterable, if True then it's (1, 1, 1),
            if False then no scaling is applied. Default True.
        cmap: object
            Matplotlib, Colorcet, cmocean, or custom colormap
        notebook: bool
            When True, the resulting plot is placed inline a jupyter notebook.
            Assumes a jupyter console is active. Automatically enables off_screen.
        backend: None or str
            Pyvista backend. Default None.
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

        if backend is not None:
            pv.set_jupyter_backend(backend)

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
            'opacity': opacity,
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
        plotter.add_slider_widget(ch_opacity, rng=slider_range, value=opacity, title='Opacity', **slider_pos)

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
            data = getattr(self.rock, attribute) if timestamp is None else getattr(self.states, attribute)[timestamp]
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
