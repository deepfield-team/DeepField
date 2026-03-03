"""Rock component."""
from typing import override
import numpy as np
import matplotlib.pyplot as plt

from .base_component import Attribute

from .base_spatial import SpatialComponent
from .utils.decorators import apply_to_each_input
from .utils.plot_utils import show_slice_static, show_slice_interactive


ROCK_ATTRIBUTES = ['PORO', 'PERMX', 'PERMY', 'PERMZ', 'KRW']


class Rock(SpatialComponent):
    """Rock component."""
    _attributes_to_load: list[Attribute] = [
        Attribute(attr, 'GRID', attr, binary_file='INIT', binary_section=attr) for attr in ROCK_ATTRIBUTES]

    @override
    @apply_to_each_input
    def to_spatial(self, attr, **kwargs):
        """Spatial order 'F' transformations."""
        _ = kwargs
        dimens = self.field.grid.dimens.values.ravel()
        self.pad_na(attr=attr)
        return self.reshape(attr=attr, newshape=dimens, order='F', inplace=True)

    @apply_to_each_input
    def pad_na(self, attr, fill_na=0., inplace=True):
        """Add dummy cells into the rock vector in the positions of non-active cells if necessary.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be padded with non-active cells.
        fill_na: float
            Value to be used as filler.
        inplace: bool
            Modify сomponent inplace.

        Returns
        -------
        output : component if inplace else padded attribute.
        """
        data = getattr(self, attr)
        dimens = self.field.grid.dimens.values.ravel()
        if np.prod(data.shape) == np.prod(dimens):
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
    def strip_na(self, attr, **kwargs):
        """Remove non-active cells from the rock vector.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be stripped

        Returns
        -------
        output : stripped attribute.
        """
        _ = kwargs
        data = self.ravel(attr)
        actnum = self.field.grid.actnum
        if data.size == np.sum(actnum):
            return data
        stripped_data = data[actnum.ravel(order='F')]
        return stripped_data

    def histogram(self, attr, **kwargs):
        """Show distribution over active cells.

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
            data = data[actnum]
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
        if np.all([i is None, j is None, k is None]):
            show_slice_interactive(self, attr, figsize=figsize, **kwargs)
        else:
            show_slice_static(self, attr, i=i, j=j, k=k, figsize=figsize, **kwargs)
        return self
