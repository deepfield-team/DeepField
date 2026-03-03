"""Base spatial component."""
from typing import override
import numpy as np

from .base_component import BaseComponent
from .utils.decorators import apply_to_each_input

class SpatialComponent(BaseComponent):
    """Base component for spatial-type attributes."""

    @apply_to_each_input
    def reshape(self, attr, newshape, order='C', inplace=True):
        """Reshape `numpy.ndarray` attributes.

        Parameters
        ----------
        attr : str, array of str
            Attribute to be reshaped.
        newshape : tuple
            New shape.
        order : str
            Numpy reshape order. Default to 'C'.
        inplace : bool
            If `True`, reshape is made inplace, return BaseComponent.
            Else, return reshaped attribute.

        Returns
        -------
        output : BaseComponent if inplace else reshaped attribute itself.
        """
        data = getattr(self, attr)
        if data is None:
            return None
        if isinstance(data, np.ndarray) and data.ndim:
            data = np.reshape(data, newshape, order=order)
        elif hasattr(data, 'reshape'):
            data = data.reshape(newshape, order=order)
        else:
            raise ValueError('Attribute {} can not be reshaped.'.format(attr))
        if inplace:
            setattr(self, attr, data)
            return self
        return data

    @apply_to_each_input
    def ravel(self, attr, order='F'):
        """Ravel attributes using Fortran order."""
        return self.reshape(attr=attr, newshape=(-1, ), order=order, inplace=False)

    @apply_to_each_input
    def to_spatial(self, attr, **kwargs):
        """Bring component to spatial state."""
        raise NotImplementedError()

    @override
    def load(self, data, binary_data, logger):
        super().load(data, binary_data, logger)
        return self.to_spatial()

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
        raise NotImplementedError()
