"""States component."""
import copy
import os
import numpy as np
import pandas as pd

from .decorators import apply_to_each_input
from .base_spatial import SpatialComponent
from .plot_utils import show_slice_static, show_slice_interactive
from .parse_utils import read_ecl_bin
from .utils import get_single_path, get_multout_paths

FULL_STATE_KEYS = ('PRESSURE', 'RS', 'SGAS', 'SOIL', 'SWAT')


class States(SpatialComponent):
    """States component of geological model."""

    def __init__(self, *args, **kwargs):
        self._dates = kwargs['dates'] if 'dates' in kwargs else pd.to_datetime([])
        super().__init__(*args, **{k:v for k, v in kwargs.items() if k != 'dates'})

    def copy(self):
        states_copy = super().copy()
        states_copy.dates = copy.deepcopy(self.dates)
        return states_copy

    def _dump_hdf5_group(self, grp, compression, state, **kwargs):
        super()._dump_hdf5_group(grp, compression, state, **kwargs)
        grp = grp[self.class_name] if self.class_name in grp else grp.create_group(self.class_name)
        grp.attrs['DATES'] = self.dates.astype(np.int64)

    @property
    def dates(self):
        """Dates for which states are available.

        Returns
        -------
        pandas.DatetimeIndex
            Dates.
        """
        return self._dates

    @dates.setter
    def dates(self, value):
        self._dates = value

    @property
    def n_timesteps(self):
        """Effective number of timesteps."""
        if not self.attributes:
            return 0
        return np.min([x.shape[0] for _, x in self.items()])

    @apply_to_each_input
    def apply(self, func, attr, *args, inplace=False, **kwargs):
        """Apply function to each timestamp of states attributes.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept data as its first argument.
        attr : str, array-like
            Attributes to get data from.
        args : misc
            Any additional positional arguments to ``func``.
        inplace: bool
            Modify сomponent inplace.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        output : States
            Transformed component.
        """
        data = getattr(self, attr)
        res = np.array([func(x, *args, **kwargs) for x in data])
        if inplace:
            setattr(self, attr, res)
            return self
        return res

    @apply_to_each_input
    def _to_spatial(self, attr):
        """Spatial order 'F' transformations."""
        dimens = self.field.grid.dimens
        self.pad_na(attr=attr)
        return self.reshape(attr=attr, newshape=(-1,) + tuple(dimens),
                            order='F', inplace=True)

    @apply_to_each_input
    def _ravel(self, attr):
        """Ravel order 'F' transformations."""
        return self.reshape(attr=attr, newshape=(self.n_timesteps, -1), order='F', inplace=False)

    @apply_to_each_input
    def pad_na(self, attr, fill_na=0., inplace=True):
        """Add dummy cells into the state vector in the positions of non-active cells if necessary.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be padded with non-active cells.
        actnum: array-like of type bool
            Vector representing a mask of active and non-active cells.
        fill_na: float
            Value to be used as filler.
        inplace: bool
            Modify сomponent inplace.

        Returns
        -------
        output : component if inplace else padded attribute.
        """
        data = getattr(self, attr)
        if np.prod(data.shape[1:]) == np.prod(self.field.grid.dimens):
            return self if inplace else data
        actnum = self.field.grid.actnum

        if data.ndim > 2:
            raise ValueError('Data should be raveled before padding.')
        n_ts = data.shape[0]

        actnum_ravel = actnum.ravel(order='F').astype(bool)
        not_actnum_ravel = ~actnum_ravel
        padded_data = np.empty(shape=(n_ts, actnum.size), dtype=float)
        padded_data[..., actnum_ravel] = data
        del data
        padded_data[..., not_actnum_ravel] = fill_na

        if inplace:
            setattr(self, attr, padded_data)
            return self
        return padded_data

    @apply_to_each_input
    def strip_na(self, attr):
        """Remove non-active cells from the state vector.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be stripped
        actnum: array-like of type bool
            Vector representing mask of active and non-active cells.

        Returns
        -------
        output : stripped attribute.

        Notes
        -----
        Outputs 1d array for each timestamp.
        """
        data = self.ravel(attr)
        actnum = self.field.grid.actnum
        if data.shape[1] == np.sum(actnum):
            return data
        stripped_data = data[..., actnum.ravel(order='F')]
        return stripped_data

    def __getitem__(self, keys):
        if isinstance(keys, str):
            return super().__getitem__(keys)
        out = self.__class__()
        for attr, data in self.items():
            data = data[keys].reshape((-1,) + data.shape[1:])
            setattr(out, attr, data)
        out.set_state(**self.state.as_dict())
        return out

    def show_slice(self, attr, t=None, i=None, j=None, k=None, figsize=None, **kwargs):
        """Visualize slices of 4D states arrays. If no slice is specified, spatial slices
        will be shown with interactive slider widgets.

        Parameters
        ----------
        attr : str
            Attribute to show.
        t : int or None, optional
            Timestamp to show.
        i : int or None, optional
            Slice along x-axis to show.
        j : int or None, optional
            Slice along y-axis to show.
        k : int or None, optional
            Slice along z-axis to show.
        figsize : array-like, optional
            Output plot size.
        kwargs : dict, optional
            Additional keyword arguments for plot.
        """
        if np.all([t is None, i is None, j is None, k is None]):
            show_slice_interactive(self, attr, figsize=figsize, **kwargs)
        else:
            show_slice_static(self, attr, t=t, i=i, j=j, k=k, figsize=figsize, **kwargs)
        return self

    def _read_buffer(self, path_or_buffer, attr, **kwargs):
        super()._read_buffer(path_or_buffer, attr, **kwargs)
        return self.reshape(attr=attr, newshape=(1, -1))

    def _load_ecl_binary(self, path_to_results, attrs, basename, logger=None, **kwargs):
        """Load states from binary ECLIPSE results files.

        Parameters
        ----------
        path_to_results : str
            Path to the folder with precomputed results of hydrodynamical simulation
        attrs : list or str
            Keyword names to be loaded
        logger : logger
            Logger for messages.
        **kwargs : dict, optional
            Any kwargs to be passed to load method.

        Returns
        -------
        states : States
            States with loaded attributes.
        """
        if attrs is None:
            attrs = FULL_STATE_KEYS

        rsspec_path = get_single_path(path_to_results, basename + '.RSSPEC')
        if rsspec_path is not None:
            self._load_ecl_rsspec(rsspec_path, logger=logger)
        unifout_path = get_single_path(path_to_results, basename + '.UNRST', logger)
        multout_paths = get_multout_paths(path_to_results, basename)

        if unifout_path is not None:
            return self._load_ecl_bin_unifout(unifout_path, attrs=attrs, logger=logger, **kwargs)
        if multout_paths is not None:
            return self._load_ecl_bin_multout(multout_paths, attrs=attrs, logger=logger, **kwargs)
        if logger is not None:
            logger.warning('The results in "%s" were not found!' % path_to_results)
            return self
        raise FileNotFoundError('The results in "%s" were not found!' % path_to_results)

    def _load_ecl_rsspec(self, path, logger, subset=None, **kwargs):
        _ = kwargs
        data = read_ecl_bin(path, attrs=['TIME', 'ITIME'], sequential=True, subset=subset,
                            logger=logger)
        dates = pd.to_datetime([])
        if len(data['ITIME'][0]) > 1:
            dates = pd.to_datetime(
                [f'{row[3]}-{row[2]}-{row[1]}' for row in data['ITIME']]
            )
        else:
            dates = pd.DatetimeIndex(
                [pd.to_datetime(self.field.meta['START']) +
                 pd.Timedelta(v[0], 'days') for v in data['TIME']])
        self._dates = dates
        return self

    def _load_hdf5_group(self, grp, attrs, raise_errors, logger, subset):
        grp_tmp = grp[self.class_name]
        if 'DATES' in grp.attrs:
            self._dates = pd.to_datetime(grp_tmp.attrs['DATES'])
        return super()._load_hdf5_group(grp, attrs, raise_errors, logger, subset)

    def _load_ecl_bin_unifout(self, path, attrs, logger, subset=None, **kwargs):
        """Load states from .UNRST binary file.

        Parameters
        ----------
        path: str
            Path to the .UNRST file.
        attrs: list or str
            Keyword names to be loaded from the file.
        kwargs : dict, optional
            Any kwargs to be passed to load method.

        Returns
        -------
        states : States
            States with loaded attributes.
        """
        _ = kwargs
        if isinstance(attrs, str):
            attrs = [attrs]
        states = read_ecl_bin(path, attrs, sequential=True, subset=subset, logger=logger)
        for attr, x in states.items():
            setattr(self, attr, np.array(x))
            self.state.binary_attributes.append(attr)
        return self

    def _load_ecl_bin_multout(self, paths, attrs, logger, subset=None, **kwargs):
        """Load states from .X____ binary files.

        Parameters
        ----------
        paths: list
            List of paths to .X____ files
        attrs: list or str
            Keyword names to be loaded from the files.
        kwargs : dict, optional
            Any kwargs to be passed to load method.

        Returns
        -------
        states : States
            States with loaded attributes.
        """
        _ = kwargs
        if isinstance(attrs, str):
            attrs = [attrs]
        states = {}
        logger.info('Start reading X files.')

        def is_in_subset(x):
            fmt = os.path.splitext(x)[1]
            timestep = int(fmt.lstrip('.X'))
            criteria = timestep in subset
            return criteria

        paths = filter(is_in_subset if subset is not None else None, paths)
        for path in paths:
            state = read_ecl_bin(path, attrs, logger=logger)
            for attr, x in state.items():
                if attr not in states:
                    states[attr] = [x]
                else:
                    states[attr].append(x)
        logger.info('Finish reading X files.')
        states = {attr: np.stack(x) for attr, x in states.items()}
        for attr, x in states.items():
            setattr(self, attr, x)
            self.state.binary_attributes.append(attr)
        return self

    def _make_data_dump(self, attr, fmt=None, actnum=None, float_dtype=None, **kwargs):
        """Prepare data for dump."""
        if fmt.upper() == 'ASCII':
            data = self.ravel(attr=attr)
            return data[0]
        if fmt.upper() == 'HDF5':
            if attr.upper() == 'DATES':
                return self.dates.astype(np.int64)
            if actnum is None:
                data = self.ravel(attr=attr)
            else:
                data = self.strip_na(attr=attr)
            return data if float_dtype is None else data.astype(float_dtype)
        return super()._make_data_dump(attr, fmt=fmt, **kwargs)
