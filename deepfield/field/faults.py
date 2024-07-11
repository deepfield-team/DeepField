#pylint: disable=too-many-lines
"""faults and FaultSegment components."""
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from anytree import (RenderTree, AsciiStyle, Resolver, PreOrderIter, PostOrderIter,
                     find_by_attr)
from itertools import product

from .parse_utils.ascii import INT_NAN

from .fault_segment import FaultSegment
from .rates import calculate_cf, show_rates, show_rates2, show_blocks_dynamics
from .grids import OrthogonalUniformGrid
from .base_component import BaseComponent
from .utils import full_ind_to_active_ind, active_ind_to_full_ind
from .wells_load_utils import load_rsm, load_ecl_binary, DEFAULTS, VALUE_CONTROL
#from .getting_faultblocks import defining_faultblocks_vtk, find_first_entering_point, defining_faultblocks_compdat
from .faults_load_utils import load_faults, load_multflt
from .decorators import apply_to_each_segment, state_check


class IterableFaults:
    """Faults iterator."""
    def __init__(self, root):
        self.iter = PreOrderIter(root)

    def __next__(self):
        x = next(self.iter)
        if x.ntype != 'fault':
            return next(self)
        return x


class Faults(BaseComponent):
    """Faults component.

    Contains faults and groups in a single tree structure, faults attributes
    and preprocessing actions.

    Parameters
    ----------
    node : FaultSegment, optional
        Root node for fault's tree.
    """

    def __init__(self, node=None, **kwargs):
        super().__init__(**kwargs)
        self._root = FaultSegment(name='FIELD', ntype="group") if node is None else node
        self._resolver = Resolver()
        self.init_state(has_blocks=False,
                        has_cf=False,
                        spatial=True,
                        all_tracks_complete=False,
                        all_tracks_inside=False,
                        full_perforation=False) #?

    def copy(self):
        """Returns a deepcopy. Cached properties are not copied."""
        copy = self.__class__(self.root.copy())
        copy._state = deepcopy(self.state) #pylint: disable=protected-access
        for node in PreOrderIter(self.root):
            if node.is_root:
                continue
            node_copy = node.copy()
            node_copy.parent = copy[node.parent.name]
        return copy

    @property
    def root(self):
        """Tree root."""
        return self._root

    @property
    def resolver(self):
        """Tree resolver."""
        return self._resolver #?

    @property
    def main_branches(self):
        """List of main branches names."""
        return [node.name for node in self if node.is_main_branch] #?

    @property
    def names(self):
        """List of fault names."""
        return [node.name for node in self] #?

    def __getitem__(self, key):
        node = find_by_attr(self.root, key)
        if node is None:
            raise KeyError(key)
        return node

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        self.drop(key)

    def __iter__(self):
        return IterableFaults(self.root)

    def __contains__(self, key):
        return find_by_attr(self.root, key) is not None

    #?
    def _get_fmt_loader(self, fmt):
        """Get loader for given file format."""
        if fmt == 'RSM':
            return self._load_rsm
        return super()._get_fmt_loader(fmt)

    def update(self, faultsdata, mode='w', **kwargs):
        """Update tree nodes with new faultsdata. If node does not exists,
        it will be attached to root.

        Parameters
        ----------
        faultsdata : dict
            Keys are fault names, values are dicts with fault attributes.
        mode : str, optional
            If 'w', write new data. If 'a', try to append new data. Default to 'w'.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        out : faults
            faults with updated attributes.
        """
        def _get_parent(name, data):
            if ':' in name:
                return self[':'.join(name.split(':')[:-1])]
            if 'WELSPECS' in data:
                groupname = data['WELSPECS']['GROUP'][0]
                try:
                    return self[groupname]
                except KeyError:
                    return FaultSegment(parent=self.root, name=groupname, ntype='group')
            return self.root

        for name in sorted(faultsdata):
            data = faultsdata[name]
            name = name.strip(' \t\'"')
            try:
                node = self[name]
            except KeyError:
                parent = _get_parent(name, data)
                node = FaultSegment(parent=parent, name=name, ntype='fault')

            if 'WELSPECS' in data:
                parent = _get_parent(name, data)
                node.parent = parent

            for k, v in data.items():
                if mode == 'w':
                    setattr(node, k, v)
                elif mode == 'a':
                    if k in node.attributes:
                        att = getattr(node, k)
                        setattr(node, k, pd.concat([att, v], **kwargs))
                    else:
                        setattr(node, k, v)
                else:
                    raise ValueError("Unknown mode {}. Expected 'w' (write) or 'a' (append)".format(mode))
                att = getattr(node, k)
                if isinstance(att, pd.DataFrame) and 'DATE' in att.columns:
                    att = att.sort_values(by='DATE').reset_index(drop=True)
                    setattr(node, k, att)
        return self

    #?
    def group(self, node_names, group_name):
        """Group nodes in a new group attached to root.

        Parameters
        ----------
        node_names : array-like
            Array of nodes to be grouped.
        group_name : str
            Name of a new group.

        Returns
        -------
        out : faults
            faults with a new group added.
        """
        group = FaultSegment(name=group_name, parent=self.root, ntype="group")
        nodes = [self[name] for name in node_names]
        node_types = [node.ntype for node in nodes]
        if np.unique(node_types).size > 1:
            raise ValueError("Can not group mixture of faults and groups in a new group.")
        for node in nodes:
            node.parent = group
        return self

    def drop(self, names):#pylint:disable=arguments-renamed
        """Detach nodes by names.

        Parameters
        ----------
        names : str, array-like
            Nodes to be detached.

        Returns
        -------
        out : faults
            faults without detached nodes.
        """
        for name in np.atleast_1d(names):
            try:
                self[name].parent = None
            except KeyError:
                continue
        return self

    #?
    def drop_incomplete(self, logger=None):
        """Drop nodes with missing 'faultTRACK' and 'PERF'.

        Parameters
        ----------
        logger : logger, optional
            Logger for messages.

        Returns
        -------
        faults : faults
            faults without incomplete nodes.
        """
        for node in self:
            if not 'COMPDAT' in node.attributes or 'COMPDATL' in node.attributes:
                if not set(['faultTRACK', 'PERF']).issubset(node.attributes):
                    self.drop(node.name)
                    if logger is not None:
                        logger.info('Node %s is incomplete and is removed.' % node.name)
        self.set_state(all_tracks_complete=True)
        return self

    #?
    def drop_outside(self, keep_ancestors=False, logger=None):
        """Drop nodes with missing 'BLOCKS' (outside of the grid).

        Parameters
        ----------
        keep_ancestors : bool
            Keep all ancestors segments for a segment with nonempty 'BLOCKS'. Setting True might result
            in segments with empty 'BLOCKS', e.g. if a parent has no 'BLOCKS' but a child has nonempty
            'BLOCKS'. If False, faulttracks may be discontinued. Default False.
        logger : logger, optional
            Logger for messages.

        Returns
        -------
        faults : faults
            faults without outside nodes.
        """
        def logger_print(node):
            if logger is not None:
                logger.info(f'Segment {node.name} is outside the grid and is removed.')

        for node in PostOrderIter(self.root):
            if node.is_root and node.name == 'FIELD':
                continue
            if (node.ntype == 'fault') and (len(node.blocks) == 0):
                if node.is_leaf:
                    node.parent = None
                    logger_print(node)
                else:
                    if keep_ancestors:
                        continue
                    p = node.parent
                    p.children = list(p.children) + list(node.children)
                    node.parent = None
                    logger_print(node)
        self.set_state(all_tracks_inside=True)
        return self

    #?
    def drop_empty_groups(self, logger=None):
        """Drop groups without faults in descendants."""
        for node in PostOrderIter(self.root):
            if node.ntype != 'group':
                continue
            if not node.children:
                self.drop(node.name)
                if logger is not None:
                    logger.info(f'Group {node.name} is empty and is removed.')
        return self

    def glob(self, name):
        """Return instances at ``name`` supporting wildcards."""
        return self.resolver.glob(self.root, name)

    def render_tree(self):
        """Print tree structure."""
        print(RenderTree(self.root, style=AsciiStyle()).by_attr())
        return self

    #?
    def blocks_ravel(self, grid):
        """Transforms block coordinates into 1D representation."""
        if not self.state.spatial:
            return self
        self.set_state(spatial=False)
        return self._blocks_ravel(grid)

    #?
    def blocks_to_spatial(self, grid):
        """Transforms block coordinates into 3D representation."""
        if self.state.spatial:
            return self
        self.set_state(spatial=True)
        return self._blocks_to_spatial(grid)

    #?
    @apply_to_each_segment
    def _blocks_ravel(self, segment, grid):
        if 'BLOCKS' not in segment or not len(segment.blocks):
            return self
        res = np.ravel_multi_index(
            tuple(segment.blocks[:, i] for i in range(3)),
            dims=grid.dimens,
            order='F'
        )
        res = full_ind_to_active_ind(res, grid)
        setattr(segment, 'BLOCKS', res)
        return self

    #?
    @apply_to_each_segment
    def _blocks_to_spatial(self, segment, grid):
        if 'BLOCKS' not in segment or not len(segment.blocks):
            return self
        res = active_ind_to_full_ind(segment.blocks, grid)
        res = np.unravel_index(res, shape=grid.dimens, order='F')
        res = np.stack(res, axis=1)
        setattr(segment, 'BLOCKS', res)
        return self

    #?
    @apply_to_each_segment
    def add_faulttrack(self, segment, grid):
        """Reconstruct faulttrack from COMPDAT table.

        To connect the end point of the current segment with the start point of the next segment
        we find a set of segments with nearest start point and take a segment with the lowest depth.
        Works fine for simple trajectories only.
        Parameters
        ----------
        grid : Grid class instance
            Grid parameters.

        Returns
        -------
        comp : faults
            faults component with faultTRACK attribute added.
        """
        if ('faultTRACK' in segment) or ('COMPDAT' not in segment and 'COMPDATL' not in segment):
            return self
        if 'COMPDAT' in segment:
            df = segment.COMPDAT[['I', 'J', 'K1', 'K2']].drop_duplicates().sort_values(['K1', 'K2'])
        else:
            if (segment.COMPDATL['LGR']!='GLOBAL').any():
                raise ValueError('LGRs other than `Global` are not supported.')
            df = segment.COMPDATL[['I', 'J', 'K1', 'K2']].drop_duplicates().sort_values(['K1', 'K2'])

        i0, j0 = segment.WELSPECS[['I', 'J']].values[0]
        i0 = i0 if i0 is not None else 0
        j0 = j0 if j0 is not None else 0
        root = np.array([i0, j0, 0])
        track = []
        centroids = grid.cell_centroids
        if not grid.state.spatial:
            centroids = centroids.reshape(tuple(grid.dimens) + (3,), order='F')
        for _ in range(len(df)):
            dist = np.linalg.norm(df[['I', 'J', 'K1']] - root, axis=1)
            row = df.iloc[[dist.argmin()]]
            track.append(centroids[int(row['I'])-1, int(row['J'])-1, int(row['K1'])-1])
            track.append(centroids[int(row['I'])-1, int(row['J'])-1, int(row['K2'])-1])
            root = row[['I', 'J', 'K2']].values.astype(float).ravel()
            df = df.drop(row.index)
        track = pd.DataFrame(track).drop_duplicates().values
        segment.faultTRACK = np.concatenate([track, np.full((len(track), 1), np.nan)], axis=1)
        return self

    #?
    @apply_to_each_segment
    def get_faultblocks(self, segment, grid, logger=None, **kwargs):
        """Calculate grid blocks for the tree of faults.

        Parameters
        ----------
        grid : class instance
            Basic grid class.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        comp : faults
            faults component with calculated grid blocks and fault in block projections.
        """
        blocks_fault = []
        for idx in segment.faults.index:
            cells = segment.faults.loc[idx, ['IX1', 'IX2', 'IY1', 'IY2', 'IZ1', 'IZ2']]
            X_range = range(cells['IX1']-1, cells['IX2']-1)
            Y_range = range(cells['IY1']-1, cells['IY2']-1)
            Z_range = range(cells['IZ1']-1, cells['IZ2']-1)
            blocks_segment = list(product(X_range, Y_range, Z_range))
            blocks_fault.extend(blocks_segment)
            
        segment.blocks = np.array(blocks_fault)
        segment.blocks_xyz = grid.xyz[segment.blocks[:, 0], 
                                      segment.blocks[:, 1], 
                                      segment.blocks[:, 2]]

        # self.set_state(has_blocks=True,
        #               has_cf=False,
        #               all_tracks_inside=False,
        #               full_perforation=False)
        return self

    #?
    @state_check(lambda state: state.full_perforation)
    @apply_to_each_segment
    def compute_events(self, segment, grid, attr='EVENTS'):
        """Make events from WCONPROD and WCONINJE if present.

        Parameters
        ----------
        grid : Grid
            Grid component.
        attr : str, optional
            Attribute name for events. Default to 'EVENTS'.

        Returns
        -------
        faults : faults
            faults with an event attribute added.
        """
        if 'wconprod' not in segment and 'wconinje' not in segment:
            return self
        df = pd.DataFrame(columns=['DATE', 'MODE', 'DREF', 'GIT', 'WIT', 'BHPT', 'LPT'])
        if segment.perforated_blocks().size == 0:
            setattr(segment, attr, df)
            return self
        wconprod = segment.wconprod if 'wconprod' in segment else pd.DataFrame()
        wconinje = segment.wconinje if 'wconinje' in segment else pd.DataFrame()
        wcon = pd.concat([wconprod, wconinje])
        if wcon.empty:
            setattr(segment, attr, df)
            return self
        wcon = wcon.sort_values('DATE').reset_index(drop=True)
        df['DATE'] = wcon['DATE']
        dref_index = np.argmin(segment.perforated_blocks()[:, 2])
        dref_index = segment.perforated_blocks()[:, :3][dref_index]
        df['DREF'] = grid.cell_centroids[dref_index[0], dref_index[1], dref_index[2]][2]

        df['MODE'] = wcon['MODE']
        mask = wcon['CONTROL'] == 'RATE'
        df.loc[mask, 'MODE'] = df.loc[mask, 'MODE'].replace('OPEN', 'INJE')
        df.loc[~mask, 'MODE'] = df.loc[~mask, 'MODE'].replace('OPEN', 'PROD')
        df['BHPT'] = wcon['BHPT']
        if 'LPT' in wcon:
            df['LPT'] = wcon['LPT']
        if 'PHASE' in wcon:
            mask = wcon['PHASE'] == 'WATER'
            df.loc[mask, 'WIT'] = wcon.loc[mask, 'SPIT']
            mask = wcon['PHASE'] == 'GAS'
            df.loc[mask, 'GIT'] = wcon.loc[mask, 'SPIT']

        setattr(segment, attr, df)
        return self

    #?
    @state_check(lambda state: state.full_perforation)
    @apply_to_each_segment
    def results_to_events(self, segment, grid, production_mode='BHPT', attr='EVENTS',
                          drop_duplicates=True):
        """Make events from results.

        Parameters
        ----------
        grid : Grid
            Grid component.
        production_mode: str, 'BHPT' or 'LPT'. Default to 'BHPT'.
            Control mode for production faults.
        attr : str, optional
            Attribute name for events. Default to 'EVENTS'.
        drop_duplicates : bool, optional
            Drop repeated events. Default to True.

        Returns
        -------
        faults : faults
            faults with an event attribute added.
        """
        if 'RESULTS' not in segment.attributes:
            return self
        df = pd.DataFrame(columns=['DATE', 'MODE', 'DREF', 'WIT', 'BHPT', 'LPT', 'GIT'])
        if segment.perforated_blocks().size == 0 or segment.results.empty:
            setattr(segment, attr, df)
            return self

        results = segment.results.sort_values('DATE')
        results['DATE'] = results['DATE'].dt.date
        results = results.drop_duplicates(subset='DATE', keep='last', ignore_index=True)
        df['DATE'] = results['DATE'][:-1]
        results = results[1:].reset_index(drop=True)
        dref_index = np.argmin(segment.perforated_blocks()[:, 2])
        dref_index = segment.perforated_blocks()[:, :3][dref_index]
        df['DREF'] = grid.cell_centroids[dref_index[0], dref_index[1], dref_index[2]][2]

        if 'WWIR' in results:
            mask_inje_water = (results['WWIR'] > 0).values
            df.loc[mask_inje_water, 'WIT'] = results.loc[mask_inje_water, 'WWIR']
            df.loc[mask_inje_water, 'MODE'] = 'INJE'
        else:
            mask_inje_water = np.zeros(df['DATE'].shape, bool)

        if 'WGIR' in results:
            mask_inje_gas = (results['WGIR'] > 0).values
            df.loc[mask_inje_gas, 'GIT'] = results.loc[mask_inje_gas, 'WGIR']
            df.loc[mask_inje_gas, 'MODE'] = 'INJE'
        else:
            mask_inje_gas = np.zeros(df['DATE'].shape, bool)

        if 'WBHP' in results and production_mode == 'BHPT':
            mask_prod = (results['WBHP'] > 0).values & ~mask_inje_water & ~mask_inje_gas & (
                (results.get('WOPR', 0) > 0) |
                (results.get('WWPR', 0) > 0) |
                (results.get('WGPR', 0) > 0)).values
            df.loc[mask_prod, 'BHPT'] = results.loc[mask_prod, 'WBHP']
            df.loc[mask_prod, 'MODE'] = 'PROD'
        elif set(('WBHP', 'WOPR', 'WWPR')).issubset(set(results)) and production_mode == 'LPT':
            mask_prod = (results['WBHP'] > 0).values & ~mask_inje_water & (
                (results.get('WOPR', 0) > 0) |
                (results.get('WWPR', 0) > 0) |
                (results.get('WGPR', 0) > 0)).values
            df.loc[mask_prod, 'LPT'] = results.loc[mask_prod, ['WOPR', 'WWPR']].sum(axis=1)
            df.loc[mask_prod, 'MODE'] = 'PROD'
        else:
            mask_prod = False
        mask_stop = ~mask_prod & ~mask_inje_water & ~mask_inje_gas
        df.loc[mask_stop, 'MODE'] = 'STOP'

        if drop_duplicates:
            arr = df.values.astype('str')
            ind_to_drop = np.where((arr[:-1, 1:] == arr[1:, 1:]).all(axis=1))[0] + 1
            df = df.drop(index=ind_to_drop).reset_index(drop=True)
        setattr(segment, attr, df)

        if 'COMPDAT' not in segment and 'COMPDATL' not in segment:
            return self

        columns = ['DATE', 'fault', 'MODE', 'CONTROL',
                   'OPT', 'WPT', 'GPT', 'SLPT', 'LPT', 'BHPT']

        wconprod = pd.DataFrame(columns=columns)

        mask_prod = (df['MODE'] != 'INJE').values
        prod = df[mask_prod]
        wconprod['DATE'] = prod['DATE']
        wconprod['fault'] = segment.name
        wconprod['MODE'] = prod['MODE']
        wconprod.loc[wconprod['MODE'] != 'STOP', 'MODE'] = 'OPEN'

        if production_mode == 'BHPT':
            wconprod['CONTROL'] = 'BHP'
            wconprod['BHPT'] = prod['BHPT']
        elif production_mode == 'LPT':
            wconprod['CONTROL'] = 'LRAT'
            wconprod['LPT'] = prod['LPT']
        else:
            raise ValueError('Unknown production mode {}'.format(production_mode))

        columns = ['DATE', 'fault', 'PHASE', 'MODE', 'CONTROL', 'SPIT', 'PIT', 'BHPT']
        wconinje = pd.DataFrame(columns=columns)
        mask_inje = (df['MODE'] == 'INJE').values
        inje = df[mask_inje]
        wconinje['DATE'] = inje['DATE']
        wconinje['fault'] = segment.name
        wconinje['MODE'] = 'OPEN'
        wconinje['CONTROL'] = 'RATE'
        wconinje.loc[inje['GIT'] > 0, 'PHASE'] = 'GAS'
        wconinje.loc[inje['WIT'] > 0, 'PHASE'] = 'WATER'
        wconinje.loc[wconinje['PHASE'] == 'GAS', 'SPIT'] = inje['GIT']
        wconinje.loc[wconinje['PHASE'] == 'WATER', 'SPIT'] = inje['WIT']

        first_date = (segment.compdat['DATE'].dropna().min() if 'COMPDAT' in segment.attributes else
                      segment.compdatl['DATE'].dropna().min())
        wconprod = wconprod.loc[wconprod['DATE'] >= first_date].reset_index(drop=True)
        wconinje = wconinje.loc[wconinje['DATE'] >= first_date].reset_index(drop=True)

        setattr(segment, 'WCONPROD', wconprod)
        setattr(segment, 'WCONINJE', wconinje)
        return self

    #?
    @apply_to_each_segment
    def calculate_cf(self, segment, rock, grid, beta=1, units='METRIC', cf_aggregation='sum', **kwargs):
        """Calculate connection factor values for each grid block of a segment.

        Parameters
        ----------
        rock : Rock class instance
            Rock component of geological model.
        grid : Grid class instance
            Rock component of geological model.
        segment : FaultSegment class instance
            fault's node.
        beta : list or ndarray
            Additional multiplicator for productivity index of fault.
        conversion_const : float
            Сonversion factor for field units.

        Returns
        -------
        comp : faults
            faults component with a 'CF' columns in a blocks_info attribute.
        """
        calculate_cf(rock, grid, segment, beta, units, cf_aggregation, **kwargs)
        self.set_state(has_cf=True)
        return self

    #?
    @apply_to_each_segment
    def apply_perforations(self, segment, current_date=None):
        """Open or close perforation intervals for given time interval.

        Parameters
        ----------
        segment : FaultSegment
            fault's node.
        current_date : pandas.Timestamp, optional
            Final date to open new perforations.

        Returns
        -------
        comp : faults
            faults component with an updated blocks_info attribute which contains:
            - projections of faulttrack in grid blocks
            - MD values for corresponding grid blocks
            - ratio of perforated part of the fault for each grid block.
        """
        segment.apply_perforations(current_date)
        self.set_state(full_perforation=current_date is None)
        return self
    
    #?
    @apply_to_each_segment
    def update_wpi_mult(self, segment, mults):
        """Update WPI multipliers.

        Parameters
        ----------
        segment : FaultSegment
            fault node
        mults : pandas.DataFrame
            Multiplyers to be updated.
            A table with following columns
            -- I:
            -- J:
            -- K: indices
            -- MULT correspnding multipliers

        Returns
        -------
        faults
            faults component with updated WPI multipliers in `perf` tables.
        """
        res = pd.merge(segment.perf, mults, on=['I', 'J', 'K'], how='left')
        res['MULT'] = res['MULT_y'].fillna(res['MULT_x'])
        res = res.drop(['MULT_x', 'MULT_y'], axis=1)
        segment.perf = res
        return self

    #?
    @apply_to_each_segment
    def filter_blocks_with_perm(self, segment, permeab):
        """Delete blocks with zero permeability."""
        x_blocks, y_blocks, z_blocks = segment.blocks.T
        permx = permeab[0, x_blocks, y_blocks, z_blocks]
        permy = permeab[1, x_blocks, y_blocks, z_blocks]
        permz = permeab[2, x_blocks, y_blocks, z_blocks]
        ind_del = list(set(np.where(permx == 0)[0]).union(
            np.where(permy == 0)[0]).union(np.where(permz == 0)[0]))
        segment.blocks = np.delete(segment.blocks, ind_del, 0)
        segment.blocks_info = segment.blocks_info.drop(ind_del, axis=0)
        return self

    def show_faults(self, figsize=None, c='r', **kwargs):
        """Return 3D visualization of faults.

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
            arr = segment.faulttrack
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=c, **kwargs)
            ax.text(*arr[0, :3], s=segment.name)

        ax.invert_zaxis()
        ax.view_init(azim=60, elev=30)

    #?
    def show_blocks_dynamics(self, timesteps=None, faultnames=None, figsize=(16, 6)):
        """Plot liquid or gas rates and pvt props for a chosen block of
        a chosen fault segment on two separate axes.

        Parameters
        ----------
        timesteps : list of Timestamps
            Dates at which rates were calculated.
        faultnames : array-like
            List of faults to plot.
        figsize : tuple
            Figsize for two axes plots.
        """
        timesteps = self.result_dates if timesteps is None else timesteps
        faultnames = self.names if faultnames is None else faultnames
        return show_blocks_dynamics(self, timesteps=timesteps, faultnames=faultnames, figsize=figsize)

    #?
    def _load_hdf5(self, path, attrs=None, **kwargs):
        """Load data from a HDF5 file.

        Parameters
        ----------
        path : str
            Path to file to load data from.
        attrs : str or array of str, optional
            Array of dataset's names to get from file. If not given, loads all.

        Returns
        -------
        comp : BaseComponent
            BaseComponent with loaded attributes.
        """
        _ = kwargs
        if isinstance(attrs, str):
            attrs = [attrs]

        def update_tree(grp, parent=None):
            """Build tree recursively following HDF5 node hierarchy."""
            if parent is None:
                node = self.root
            else:
                ntype = grp.attrs.get('ntype', None)
                if ntype is None: #backward compatibility, will be removed in a future
                    ntype = 'group' if grp.attrs.get('is_group', False) else 'fault'
                node = FaultSegment(parent=parent, name=grp.name.split('/')[-1], ntype=ntype)
            for k, v in grp.items():
                if k == 'data':
                    for att in v.keys() if attrs is None else attrs:
                        try:
                            data = v[att]
                        except KeyError:
                            continue
                        if isinstance(data, h5py.Group):
                            data = pd.read_hdf(path, key='/'.join([grp.name, 'data', att]), mode='r')
                            setattr(node, att, data)
                        else:
                            setattr(node, att, data[()])
                else:
                    update_tree(v, node)

        with h5py.File(path, 'r') as f:
            self.set_state(**dict(f[self.class_name].attrs.items()))
            update_tree(f[self.class_name])
        return self

    #?
    def _read_buffer(self, buffer, attr, **kwargs):
        """Load fault data from an ASCII file.

        Parameters
        ----------
        buffer : StringIteratorIO
            Buffer to get string from.
        attr : str
            Target keyword.

        Returns
        -------
        comp : faults
            faults component with loaded fault data.
        """
        if attr == 'FAULTS':
            return load_faults(self, buffer, **kwargs)
        if attr == 'MULTFLT':
            return load_multflt(self, buffer, **kwargs)
        raise ValueError("Keyword {} is not supported in faults.".format(attr))

    #?
    def _load_rsm(self, *args, **kwargs):
        """Load RSM fault data from file."""
        return load_rsm(self, *args, **kwargs)

    #?
    def _load_ecl_binary(self, *args, **kwargs):
        """Load results from UNSMRY file."""
        return load_ecl_binary(self, *args, **kwargs)

    #?
    def _dump_hdf5(self, path, mode='a', state=True, **kwargs):  #pylint: disable=too-many-branches
        """Save data into HDF5 file.

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
            Default to 'a'.
        state : bool
            Dump compoments's state.

        Returns
        -------
        comp : faults
            faults unchanged.
        """
        _ = kwargs
        with h5py.File(path, mode) as f:
            faults = f[self.class_name] if self.class_name in f else f.create_group(self.class_name)
            if state:
                for k, v in self.state.as_dict().items():
                    faults.attrs[k] = v
            for node in PreOrderIter(self.root):
                if node.is_root:
                    continue
                node_path = node.fullname
                if node.name == 'data':
                    raise ValueError("Name 'data' is not allowed for nodes.")
                grp = faults[node_path] if node_path in faults else faults.create_group(node_path)
                grp.attrs['ntype'] = node.ntype
                if 'data' not in grp:
                    grp_data = faults.create_group(node_path + '/data')
                else:
                    grp_data = grp['data']
                for att, data in node.items():
                    if isinstance(data, pd.DataFrame):
                        continue
                    if att in grp_data:
                        del grp_data[att]
                    grp_data.create_dataset(att, data=data)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for node in PreOrderIter(self.root):
                if node.is_root:
                    continue
                for att, data in node.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_hdf(path, key='/'.join([self.class_name, node.fullname,
                                                        'data', att]), mode='a')
        return self


    @apply_to_each_segment
    def fill_na(self, segment, attr):
        """
        Fill ionvalid values in faults segment attribute.

        Parameters
        ----------
        attr: str
            Attribute name.

        Returns
        -------
        comp : faults
            faults with fixed attribute.
        """
        if attr in segment.attributes:
            data = getattr(segment, attr)
            welspecs = segment.welspecs
            if set(('I', 'J')).issubset(set(data.columns)):
                data['I'] = data['I'].replace(INT_NAN, welspecs['I'].values[0])
                data['J'] = data['J'].replace(INT_NAN, welspecs['J'].values[0])
        return self

    def _dump_ascii(self, path, attr, mode='w', **kwargs):
        """Save data into text file.

        Parameters
        ----------
        path : str
            Path to output file.
        attr : str
            Attribute to dump into file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'w'.

        Returns
        -------
        comp : faults
            faults unchanged.

        with open(path, mode) as f:
            if attr.upper() == 'faultTRACK':
                for node in self:
                    if 'faultTRACK' in node and 'COMPDAT' not in node and 'COMPDATL' not in node:
                        f.write('faultTRACK\t{}\n'.format(node.name))
                        for line in node.faulttrack:
                            f.write(' '.join(line.astype(str)) + '\n')
            elif attr.upper() == 'PERF':
                write_perf(f, self, DEFAULTS)
            elif attr.upper() == 'GROUP':
                for node in PreOrderIter(self.root):
                    if node.is_root:
                        continue
                    if node.ntype == 'group' and not node.is_leaf and not node.children[0].ntype == 'group':
                        f.write(' '.join(['GROUP', node.name] +
                                         [child.name for child in node.children]) + '\n')
                f.write('/\n')
            elif attr.upper() == 'GRUPTREE':
                f.write('GRUPTREE\n')
                for node in PreOrderIter(self.root):
                    if node.is_root:
                        continue
                    if node.ntype == 'group' and node.parent.ntype == 'group':
                        p_name = '1*' if node.parent.is_root else node.parent.name
                        f.write(' '.join([node.name, p_name, '/\n']))
                f.write('/\n')
            elif attr.upper() == 'EVENTS':
                write_events(f, self, VALUE_CONTROL)
            elif attr.upper() == 'SCHEDULE':
                write_schedule(f, self, **kwargs)
            elif attr.upper() == 'WELSPECS':
                write_welspecs(f, self)
            else:
                raise NotImplementedError("Dump for {} is not implemented.".format(attr.upper()))        """

    @apply_to_each_segment
    def _get_first_entering_point(self, segment, grid, **kwargs):
        """Calculate grid blocks for the tree of faults.

        Parameters
        ----------
        grid : Grid
            Basic grid class.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        comp : faults
            faults component with calculated grid blocks and fault in block projections.
        """
        _ = kwargs
        grid = grid.as_corner_point
        if grid._vtk_locator is None or grid._cell_id_d is None: #pylint: disable=protected-access
            grid.create_vtk_locator(use_only_active=True, scaling=False)

        output = find_first_entering_point(segment.faulttrack,
                                           grid, grid._vtk_locator, grid._cell_id_d) #pylint: disable=protected-access
        segment._first_entering_point = output #pylint: disable=protected-access

        return self
