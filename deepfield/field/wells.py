"""Wells component."""
from __future__ import annotations
import logging
from typing import Self, cast, override
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets

import resdp
import resdp.binary

from .base_tree import BaseTree, BaseTreeNode
from .base_component import Attribute, T
from .grids import OrthogonalGrid

from .utils.wells_utils import load_results, load_welltrack
from .utils.grid_raycasting import get_wellblocks_vtk, get_wellblocks_compdat
from .utils.decorators import apply_to_each_node

INT_NAN = -99999999

class WellsNode(BaseTreeNode):
    """Well's node."""


class WellScheduleAttribute(Attribute[T]):
    """Well schedule attribute."""
    def __init__(self,
                 name: str | None = None,
                 kw: str | None = None,
                 custom_loader=None,
                 custom_ascii_loader=None,
                 postprocess=None,
                 not_present=None,
                 binary_file: resdp.binary.FileType | None = None,
                 binary_section=None,
                 binary_process=None,
                 sequential: bool=False,
                 dated: bool=True
                 ):
        super().__init__(name,
                         'SCHEDULE',
                         kw,
                         custom_loader,
                         custom_ascii_loader,
                         postprocess,
                         not_present,
                         binary_file,
                         binary_section,
                         binary_process,
                         sequential)
        self._dated: bool=dated

    @override
    def _load_value(self,
                    data: resdp.DataType,
                    binary_data: resdp.binary.BinaryData,
                    logger: logging.Logger | None) -> Self:
        cur_date = None
        section = cast(str, self._section)
        assert isinstance(self.component, Wells)
        res: list[pd.DataFrame] = []
        for key, val in data[section]:
            if key == 'DATES':
                assert isinstance(val, Sequence)
                assert isinstance(val[-1], pd.Timestamp)
                val = cast(Sequence[pd.Timestamp], val)
                cur_date = val[-1]
            elif key == self._kw:
                assert isinstance(val, pd.DataFrame)
                if self._dated:
                    val = val.assign(DATE=cur_date)
                res.append(val)
        if len(res) == 0:
            self._value = None
        else:
            self._value = pd.concat(res)
        return self


SIMPLE_SCHEDULE = ['WELSPECS', 'WELSPECL']
DATED_SCHEDULE = ['WCONPROD', 'WCONINJE', 'COMPDAT', 'COMPDATL', 'COMPDATMD', 'WEFAC']

class Wells(BaseTree):
    """Wells component."""
    _attributes_to_load: list[Attribute[Self]] = (
        [WellScheduleAttribute(name=attr, kw=attr, dated=False) for attr in SIMPLE_SCHEDULE] +
        [WellScheduleAttribute(name=attr, kw=attr, dated=True) for attr in DATED_SCHEDULE] +
        [Attribute(name='WELLTRACK', custom_loader=load_welltrack),
         Attribute(name='RESULTS', custom_loader=load_results)])

    def __init__(self, **kwargs):
        root = WellsNode(name='FIELD', is_group=True)
        super().__init__(root=root, **kwargs)

    def build_tree(self):
        """Build tree from component's data."""
        if 'WELSPECS' in self:
            welspecs = self.welspecs
        elif 'WELSPECL' in self:
            welspecs = self.welspecl
        else:
            return self

        groups = {'FIELD': self.root}
        for name in welspecs.GROUP.unique():
            if name in [None, 'FIELD']:
                continue
            groups[name] = WellsNode(parent=self.root, name=name, is_group=True)

        for _, row in welspecs.iterrows():
            group = 'FIELD' if row.GROUP is None else row.GROUP
            WellsNode(parent=groups[group], name=row.WELL, key='WELL')

        return self

    def add_welltrack(self, overwrite=True):
        """Cnstruct welltrack from COMPDAT table.

        To connect the end point of the current segment with the start point of the next segment
        we find a set of segments with nearest start point and take a segment with the lowest depth.
        Works fine for simple trajectories only.
        """
        if self.welltrack is not None and not overwrite: #pylint: disable=access-member-before-definition
            return self
        dfs = []
        self._get_welltrack(dfs)
        self.welltrack = pd.concat(dfs) if dfs else pd.DataFrame(columns=['X', 'Y', 'Z', 'MD']) #pylint: disable=attribute-defined-outside-init
        return self

    @apply_to_each_node
    def _get_welltrack(self, segment, dfs):
        """Construct welltrack from COMPDAT table."""
        if 'COMPDAT' not in segment and 'COMPDATL' not in segment:
            return self
        grid = self.field.grid
        if 'COMPDAT' in segment:
            df = segment.COMPDAT[['IW', 'JW', 'K1', 'K2']].drop_duplicates().sort_values(['K1', 'K2'])
        else:
            if (segment.COMPDATL['LGR']!='GLOBAL').any():
                raise ValueError('LGRs other than `Global` are not supported.')
            df = segment.COMPDATL[['IW', 'JW', 'K1', 'K2']].drop_duplicates().sort_values(['K1', 'K2'])

        i0, j0 = segment.WELSPECS[['IW', 'JW']].values[0]
        i0 = i0 if i0 is not None else 0
        j0 = j0 if j0 is not None else 0
        root = np.array([i0, j0, 0])
        track = []
        for _ in range(len(df)):
            dist = np.linalg.norm(df[['IW', 'JW', 'K1']] - root, axis=1)
            row = df.iloc[[dist.argmin()]]
            xyz = grid.get_xyz([int(row.iloc[0]['IW'])-1,
                                int(row.iloc[0]['JW'])-1,
                                int(row.iloc[0]['K1'])-1])
            track.append(xyz[:, :4].mean(axis=-2).ravel())
            xyz = grid.get_xyz([int(row.iloc[0]['IW'])-1,
                                int(row.iloc[0]['JW'])-1,
                                int(row.iloc[0]['K2'])-1])
            track.append(xyz[:, 4:].mean(axis=-2).ravel())
            root = row[['IW', 'JW', 'K2']].values.astype(float).ravel()
            df = df.drop(row.index)
        track = pd.DataFrame(track).drop_duplicates().values
        welltrack = np.concatenate([track, np.full((len(track), 1), np.nan)], axis=1)
        df = pd.DataFrame(welltrack, columns=['X', 'Y', 'Z', 'MD'])
        df['WELL'] = segment.name
        df = df[['WELL', 'X', 'Y', 'Z', 'MD']]
        dfs.append(df)
        return self

    @apply_to_each_node
    def get_blocks(self, segment: WellsNode, logger: logging.Logger | None=None):
        """Calculate grid blocks for the tree of wells."""
        compdatl_attribute = segment.compdatl
        compdat_attribute = segment.compdat

        grid = self.field.grid

        if (compdat_attribute is not None) or (compdatl_attribute is not None):
            if compdat_attribute is not None:
                compdat = compdat_attribute
            elif (cast(pd.DataFrame, compdatl_attribute)['LGR']=='GLOBAL').all():
                assert compdatl_attribute is not None
                compdat = compdatl_attribute
            else:
                if logger is not None:
                    logger.warning('Well {}: can not get blocks from COMPDATL data.'.format(segment.name))
                return self

            segment.blocks = get_wellblocks_compdat(compdat, segment.welspecs)
            blocks = cast(npt.NDArray[np.int_], segment.blocks)
            if isinstance(grid, OrthogonalGrid):
                h_well: npt.NDArray[np.float_] | npt.NDArray[np.int_] = (
                        np.stack([(0, 0, grid.dz[i[0], i[1], i[2]]) for i in blocks]))
            else:
                h_well = np.full(blocks.shape, np.NaN)
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])

        else:
            blocks, points, mds = get_wellblocks_vtk(segment.welltrack[['X', 'Y', 'Z', 'MD']].values, grid)

            segment.blocks = blocks
            h_well = abs(points[:, 1] - points[:, 0])
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])
            segment.blocks_info['MDU'] = mds[:, 0]
            segment.blocks_info['MDL'] = mds[:, 1]
            segment.blocks_info['Enter_point'] = list(points[:, 0])
            segment.blocks_info['Leave_point'] = list(points[:, 1])

        return self

    def show_wells(self, figsize=None, c='r', **kwargs):
        """Return 3D visualization of wells.

        Parameters
        ----------
        figsize : tuple
            Output figsize.
        c : str
            Line color, default red.
        kwargs : misc
            Any additional kwargs for plot.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        for segment in self:
            arr = segment.welltrack[['X', 'Y', 'Z']].values
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=c, **kwargs)
            ax.text(*arr[0, :3], s=segment.name)

        ax.invert_zaxis()
        ax.view_init(azim=60, elev=30)

    def show_rates(self, figsize=(16, 6)):
        """Plot production rates for a single well.

        Parameters
        ----------
        figsize : tuple
            Figsize for two axes plots.
        """
        rates = [x for x in self.results.columns if x not in ['DATE', 'WELL']]
        n = len(self.results['DATE'].unique())

        def update(wellname, rate, cumulative, start_step, end_step):
            rates = self[wellname].results.set_index('DATE')

            _, ax = plt.subplots(1, 1, figsize=figsize)
            title = 'Cumulative ' + rate if cumulative else rate
            ax.set_title('{} - {}'.format(wellname, title))
            ax.set_ylabel('Cumulative Rate' if cumulative else 'Rate')
            ax.set_xlabel('Date')

            data = rates[rate].cumsum() if cumulative else rates[rate]

            data.iloc[start_step:end_step].plot(ax=ax, lw=2)

        interact(update,
                 wellname=widgets.Dropdown(options=self.names),
                 rate=widgets.Dropdown(options=rates, value=rates[0]),
                 cumulative=widgets.Checkbox(value=False, description='Cumulative'),
                 start_step=widgets.IntSlider(min=0, max=n, step=1, value=0),
                 end_step=widgets.IntSlider(min=0, max=n, step=1, value=n))
        plt.show()

    def fill_nan_coordinates(self):
        """Fill nan IW and JW coordinates using WELSPECS/WELSPECL."""
        if 'WELSPECS' in self:
            welspecs = self.welspecs
        elif 'WELSPECL' in self:
            welspecs = self.welspecl
        else:
            return self

        for attr, df in self.items():
            if attr.upper() in ['WELSPECS', 'WELSPECL']:
                continue
            if set(('IW', 'JW')).issubset(set(df.columns)):
                df.loc[df.IW==INT_NAN, 'IW'] = None
                df.loc[df.JW==INT_NAN, 'JW'] = None
                df = df.set_index('WELL')
                df = df.fillna(welspecs.set_index('WELL')).reset_index()
                df[['IW', 'JW']] = df[['IW', 'JW']].astype(int)
                setattr(self, attr, df)
        return self
