"""Rock component."""
import numpy as np
import matplotlib.pyplot as plt

from .base_spatial import SpatialComponent
from .decorators import apply_to_each_input
from .plot_utils import show_slice_static, show_slice_interactive
from .utils import get_single_path
from .parse_utils import read_ecl_bin


class Rock(SpatialComponent):
    """Rock component of geological model."""

    def _load_ecl_binary(self, path_to_results, attrs, basename, logger=None):
        path = get_single_path(path_to_results, basename + '.INIT', logger)
        if path is None:
            return
        sections = read_ecl_bin(path, attrs, logger=logger)

        for k in ['PORO', 'PERMX', 'PERMY', 'PERMZ', "KRW"]:
            if (k in attrs) and (k in sections):
                setattr(self, k, sections[k])
            self.state.binary_attributes.append(k)

    @apply_to_each_input
    def _to_spatial(self, attr):
        """Spatial order 'F' transformations."""
        dimens = self.field.grid.dimens
        self.pad_na(attr=attr)
        return self.reshape(attr=attr, newshape=dimens, order='F', inplace=True)

    def _make_data_dump(self, attr, fmt=None, float_dtype=None, **kwargs):
        """Prepare data for dump."""
        if fmt.upper() != 'HDF5':
            return super()._make_data_dump(attr, fmt=fmt, **kwargs)
        data = self.ravel(attr=attr)
        return data if float_dtype is None else data.astype(float_dtype)

    @apply_to_each_input
    def pad_na(self, attr, fill_na=0., inplace=True):
        """Add dummy cells into the rock vector in the positions of non-active cells if necessary.

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
        if np.prod(data.shape) == np.prod(self.field.grid.dimens):
            return self if inplace else data
        actnum = self.field.grid.actnum
        if data.ndim > 1:
            raise ValueError('Data should be ravel for padding.')
        padded_data = np.full(shape=(actnum.size,), fill_value=fill_na, dtype=float)
        padded_data[actnum.ravel(order='F')] = data
        if inplace:
            setattr(self, attr, padded_data)
            return self
        return padded_data

    @apply_to_each_input
    def strip_na(self, attr):
        """Remove non-active cells from the rock vector.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be stripped

        Returns
        -------
        output : stripped attribute.
        """
        data = self.ravel(attr)
        actnum = self.field.grid.actnum
        if data.size == np.sum(actnum):
            return data
        stripped_data = data[actnum.ravel(order='F')]
        return stripped_data

    def show_histogram(self, attr, **kwargs):
        """Show properties distribution.

        Parameters
        ----------
        attr : str
            Attribute to compute the histogram.
        kwargs : misc
            Any additional named arguments to ``plt.hist``.

        Returns
        -------
        plot : Histogram plot.
        """
        data = getattr(self, attr)
        try:
            actnum = self.field.grid.actnum
            data = data * actnum
        except AttributeError:
            pass
        plt.hist(data.ravel(), **kwargs)
        plt.show()
        return self

    def show_slice(self, attr, i=None, j=None, k=None, figsize=None, **kwargs):
        """Visualize slices of 3D array. If no slice is specified, all 3 slices
        will be shown with interactive slider widgets.

        Parameters
        ----------
        attr : str
            Attribute to show.
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
        data = getattr(self, attr)
        try:
            actnum = self.field.grid.actnum
            data = data * actnum
        except AttributeError:
            pass
        if np.all([i is None, j is None, k is None]):
            show_slice_interactive(self, attr, figsize=figsize, **kwargs)
        else:
            show_slice_static(self, attr, i=i, j=j, k=k, figsize=figsize, **kwargs)
        return self
