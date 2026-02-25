"""Tables component."""
from __future__ import annotations
from typing import override
from collections.abc import Sequence
import numpy as np
import pandas as pd

from .base_component import Attribute, BaseComponent
from .utils.table_interpolation import TABLE_INTERPOLATOR
from .utils.plot_utils import plot_table_1d, plot_table_2d

class _Table(pd.DataFrame):  # pylint: disable=abstract-method
    """Table component."""
    _metadata = ['domain', 'name', '_interpolator']

    def __init__(self, data=None, **kwargs):
        self.name = kwargs.pop('name') if 'name' in kwargs else ''
        super().__init__(data=data, **kwargs)
        self.domain = list(self.index.names) if list(self.index.names)[0] is not None else None
        self._interpolator = None

    def __call__(self, x):
        """Apply table-defined function to x.

        Parameters
        ----------
        x: array-like of shape (n_points, len(table.domain))
            Points for function to be computed at

        Returns
        -------
        values: array-like of shape (n_points, len(table.columns))
        """
        if self._interpolator is None:
            if self.name in TABLE_INTERPOLATOR:
                self._interpolator = TABLE_INTERPOLATOR[self.name](self)
            else:
                self._interpolator = TABLE_INTERPOLATOR[None](self)
        return self._interpolator(x)

    @property
    def _constructor(self):
        return self.__class__

    def plot(self, figsize=None):
        """Plot table."""
        if self.domain:
            if len(self.domain) == 1:
                plot_table_1d(self, figsize=figsize)
            elif len(self.domain) == 2:
                plot_table_2d(self, figsize=figsize)
            else:
                raise AttributeError('Can plot functions of 1 and 2 variables. Function of %d variables is given'
                                     % len(self.domain))
        else:
            raise AttributeError('The table has no domain. Hence, can not be plotted!')

    def to_numpy(self, include_index=False):
        """
        Get numpy representation of a table.
        """
        if include_index:
            if isinstance(self.index, pd.MultiIndex):
                index = np.array(self.index.values.tolist())
            else:
                index = self.index.values.reshape(-1, 1)
            return np.hstack((index, self.values))
        return self.values


class Tables(BaseComponent):
    """Tables component of geological model."""
    _attributes_to_load: list[Attribute] = [
        Attribute(
            'SWOF',
            'PROPS',
            'SWOF'
        ),
        Attribute(
            'PVTO',
            'PROPS',
            'PVTO',
        ),
        Attribute(
            'PVTG',
            'PROPS',
            'PVTG'
        ),
        Attribute(
            'PVDG',
            'PROPS',
            'PVTG'
        ),
        Attribute(
            'PVDO',
            'PROPS',
            'PVDO'
        ),
        Attribute(
            'PVTW',
            'PROPS',
            'PVTW',
        ),
        Attribute(
            'PVCDO',
            'PROPS',
            'PVCDO'
        ),
        Attribute(
            'SGOF',
            'PROPS',
            'SGOF'
        ),
        Attribute(
            'RSVD',
            'SOLUTION',
            'RSVD'
        ),
        Attribute(
            'ROCK',
            'PROPS',
            'ROCK'
        ),
        Attribute(
            'DENSITY',
            'PROPS',
            'DENSITY'
        )
    ]

    @override
    def __getattr__(self, key) -> list[_Table] | None:
        val = super().__getattr__(key)
        if isinstance(val, Sequence) and all(isinstance(v, pd.DataFrame) for v in val):
            return [_Table(v) for v in val]
        if val is None:
            return val
        raise ValueError('Value should be a sequence of pandas DataFrames.')
