"""Wells components."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anytree import PreOrderIter, PostOrderIter

from .parse_utils.ascii import INT_NAN

from .well_segment import WellSegment
from .base_tree import BaseTree
from .rates import calculate_cf, show_rates, show_rates2, show_blocks_dynamics
from .grids import OrthogonalGrid
from .getting_wellblocks import defining_wellblocks_vtk, find_first_entering_point, defining_wellblocks_compdat
from .wells_dump_utils import write_perf, write_events, write_schedule, write_welspecs
from .wells_load_utils import (load_rsm, load_ecl_binary, load_group, load_grouptree,
                               load_welspecs, load_welspecl, load_compdat, load_compdatl,
                               load_comdatmd, load_wconprod, load_wconinje, load_welltracks,
                               load_events, load_history, load_wefac, load_wfrac, load_wfracp,
                               DEFAULTS, VALUE_CONTROL)
from .decorators import apply_to_each_segment, state_check


class Wells(BaseTree):
    """Wells component.

    Contains wells and groups in a single tree structure, wells attributes
    and preprocessing actions.

    Parameters
    ----------
    node : WellSegment, optional
        Root node for well's tree.
    """

    def __init__(self, node=None, **kwargs):
        super().__init__(node=node, nodeclass=WellSegment, **kwargs)
        self.init_state(has_blocks=False,
                        has_cf=False,
                        all_tracks_complete=False,
                        all_tracks_inside=False,
                        full_perforation=False)

    @property
    def main_branches(self):
        """List of main branches names."""
        return [node.name for node in self if node.is_main_branch]

    @property
    def event_dates(self):
        """List of dates with any event in main branches."""
        return self._collect_dates('EVENTS')

    @property
    def result_dates(self):
        """List of dates with any result in main branches."""
        return self._collect_dates('RESULTS')

    @property
    def history_dates(self):
        """List of dates with any history in main branches."""
        return self._collect_dates('HISTORY')

    def _collect_dates(self, attr):
        """List of common dates given in the attribute of main branches."""
        agg = [getattr(node, attr).DATE for node in self if node and attr in node]
        if not agg:
            return pd.to_datetime([])
        dates = sorted(pd.concat(agg).unique())
        return pd.to_datetime(dates)

    @property
    def total_rates(self):
        """Total rates over all wells."""
        return self.root.total_rates

    @property
    def cum_rates(self):
        """Cumulative rates over all wells."""
        return self.root.cum_rates

    def _get_fmt_loader(self, fmt):
        """Get loader for given file format."""
        if fmt == 'RSM':
            return self._load_rsm
        return super()._get_fmt_loader(fmt)

    def update(self, data, mode='w', **kwargs):
        """Update tree nodes with new wellsdata. If node does not exists,
        it will be attached to root.

        Parameters
        ----------
        data : dict
            Keys are well names, values are dicts with well attributes.
        mode : str, optional
            If 'w', write new data. If 'a', try to append new data. Default to 'w'.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        out : Wells
            Wells with updated attributes.
        """
        def _get_parent(name, data):
            if ':' in name:
                return self[':'.join(name.split(':')[:-1])]
            if 'WELSPECS' in data:
                groupname = data['WELSPECS']['GROUP'][0]
                try:
                    return self[groupname]
                except KeyError:
                    return WellSegment(parent=self.root, name=groupname, ntype='group', field=self.field)
            return self.root

        for name in sorted(data):
            wdata = data[name]
            name = name.strip(' \t\'"')
            try:
                node = self[name]
            except KeyError:
                parent = _get_parent(name, wdata)
                node = self._nodeclass(parent=parent, name=name, ntype='well', field=self.field)

            if 'WELSPECS' in wdata:
                parent = _get_parent(name, wdata)
                node.parent = parent

            for k, v in wdata.items():
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

    def drop_incomplete(self, logger=None, required=None):
        """Drop nodes with missing 'WELLTRACK' and 'PERF'.

        Parameters
        ----------
        logger : logger, optional
            Logger for messages.
        required: list
            Required attributes for wells. Default ['WELLTRACK', 'PERF'].

        Returns
        -------
        wells : Wells
            Wells without incomplete nodes.
        """
        if required is None:
            required = ['WELLTRACK', 'PERF']
        for node in self:
            if not (('COMPDAT' in node.attributes) or ('COMPDATL' in node.attributes)):
                if not set(required).issubset(node.attributes):
                    self.drop(node.name)
                    if logger is not None:
                        logger.info('Node %s is incomplete and is removed.' % node.name)
        self.set_state(all_tracks_complete=True)
        return self

    def drop_outside(self, keep_ancestors=False, logger=None):
        """Drop nodes with missing 'BLOCKS' (outside of the grid).

        Parameters
        ----------
        keep_ancestors : bool
            Keep all ancestors segments for a segment with nonempty 'BLOCKS'. Setting True might result
            in segments with empty 'BLOCKS', e.g. if a parent has no 'BLOCKS' but a child has nonempty
            'BLOCKS'. If False, welltracks may be discontinued. Default False.
        logger : logger, optional
            Logger for messages.

        Returns
        -------
        wells : Wells
            Wells without outside nodes.
        """
        def logger_print(node):
            if logger is not None:
                logger.info(f'Segment {node.name} is outside the grid and is removed.')

        for node in PostOrderIter(self.root):
            if node.is_root and node.name == 'FIELD':
                continue
            if (node.ntype == 'well') and (len(node.blocks) == 0):
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

    @apply_to_each_segment
    def add_welltrack(self, segment):
        """Reconstruct welltrack from COMPDAT table.

        To connect the end point of the current segment with the start point of the next segment
        we find a set of segments with nearest start point and take a segment with the lowest depth.
        Works fine for simple trajectories only.
        """
        if ('WELLTRACK' in segment) or ('COMPDAT' not in segment and 'COMPDATL' not in segment):
            return self
        grid = self.field.grid
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
        for _ in range(len(df)):
            dist = np.linalg.norm(df[['I', 'J', 'K1']] - root, axis=1)
            row = df.iloc[[dist.argmin()]]
            track.append(centroids[int(row.iloc[0]['I'])-1, int(row.iloc[0]['J'])-1, int(row.iloc[0]['K1'])-1])
            track.append(centroids[int(row.iloc[0]['I'])-1, int(row.iloc[0]['J'])-1, int(row.iloc[0]['K2'])-1])
            root = row[['I', 'J', 'K2']].values.astype(float).ravel()
            df = df.drop(row.index)
        track = pd.DataFrame(track).drop_duplicates().values
        segment.WELLTRACK = np.concatenate([track, np.full((len(track), 1), np.nan)], axis=1)
        return self

    @apply_to_each_segment
    def get_blocks(self, segment, logger=None, **kwargs):
        """Calculate grid blocks for the tree of wells.

        Parameters
        ----------
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        comp : Wells
            Wells component with calculated grid blocks and well in block projections.
        """
        grid = self.field.grid
        if 'COMPDAT' in segment.attributes:
            segment.blocks = defining_wellblocks_compdat(segment.compdat)
            segment.blocks_info = pd.DataFrame(np.empty((segment.blocks.shape[0], 0)))
            if isinstance(self.field.grid, OrthogonalGrid):
                h_well = np.stack([(0, 0, self.field.grid.dz[i[0], i[1], i[2]])
                                   for i in segment.blocks])
            else:
                h_well = np.stack([(np.NaN, np.NaN, np.NaN) for _ in range(segment.blocks.shape[0])])
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])
        elif 'COMPDATL' in segment.attributes and (segment.compdatl['LGR']=='GLOBAL').all():
            segment.blocks = defining_wellblocks_compdat(segment.compdatl)
            segment.blocks_info = pd.DataFrame(np.empty((segment.blocks.shape[0], 0)))
            h_well = np.stack([(0, 0, self.field.grid.dz) for _ in range(segment.blocks.shape[0])])
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])
        else:
            grid = grid.as_corner_point

            if grid._vtk_locator is None or grid._cell_id_d is None: #pylint: disable=protected-access
                grid.create_vtk_locator(use_only_active=True, scaling=False)

            output = defining_wellblocks_vtk(segment.welltrack, segment.name,
                                             grid, grid._vtk_locator, grid._cell_id_d, #pylint: disable=protected-access
                                             logger=logger, **kwargs)
            xyz_block, h_well, welltr_block_md, inters = output

            if len(h_well) > 0:
                if np.isclose(h_well[-1], 0).all():
                    if logger is not None:
                        logger.info((
                            'Well projections in the last block of well ' +
                            '`{}` are close to 0. Block {} will be removed '+
                            'from well blocks.'
                        ).format(segment.name, xyz_block[-1]))
                    h_well = h_well[:-1]
                    xyz_block = xyz_block[:-1]
                    welltr_block_md = welltr_block_md[:-1]
                    inters = inters[:-1]
                bases = grid.cell_bases((xyz_block[:, 0], xyz_block[:, 1], xyz_block[:, 2]))
                h_well = [np.abs(np.dot(b, h)) for b, h in zip(bases, h_well)]
            segment.blocks = xyz_block
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])
            segment._inters = inters #pylint: disable=protected-access
            segment.blocks_info['MD'] = welltr_block_md
        segment.blocks_info = segment.blocks_info.assign(
            PERF_RATIO=np.nan if len(segment.blocks_info) == 0 else 0,
            RAD=np.nan if len(segment.blocks_info) == 0 else DEFAULTS['RAD'],
            SKIN=np.nan if len(segment.blocks_info) == 0 else DEFAULTS['SKIN'],
            MULT=np.nan if len(segment.blocks_info) == 0 else DEFAULTS['MULT'],
        )

        self.set_state(has_blocks=True,
                       has_cf=False,
                       all_tracks_inside=False,
                       full_perforation=False)
        return self

    @apply_to_each_segment
    def add_null_results_column(self, segment, column, fill_with=0.):
        """Add a dummy column to results if it is absent.

        Parameters
        ----------
        segment
        column: str
            Column name to add
        fill_with: float
            Value to fill the column

        Returns
        -------
        wells: Wells
        """
        if 'RESULTS' not in segment.attributes:
            return self
        if (segment.perforated_blocks().size != 0) and column not in segment.results:
            segment.results[column] = fill_with
        return self

    @state_check(lambda state: state.full_perforation)
    @apply_to_each_segment
    def compute_events(self, segment, attr='EVENTS'):
        """Make events from WCONPROD and WCONINJE if present.

        Parameters
        ----------
        attr : str, optional
            Attribute name for events. Default to 'EVENTS'.

        Returns
        -------
        wells : Wells
            Wells with an event attribute added.
        """
        if 'wconprod' not in segment and 'wconinje' not in segment:
            return self
        grid = self.field.grid
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

    @state_check(lambda state: state.full_perforation)
    @apply_to_each_segment
    def results_to_events(self, segment, production_mode='BHPT', attr='EVENTS',
                          drop_duplicates=True):
        """Make events from results.

        Parameters
        ----------
        production_mode: str, 'BHPT' or 'LPT'. Default to 'BHPT'.
            Control mode for production wells.
        attr : str, optional
            Attribute name for events. Default to 'EVENTS'.
        drop_duplicates : bool, optional
            Drop repeated events. Default to True.

        Returns
        -------
        wells : Wells
            Wells with an event attribute added.
        """
        if 'RESULTS' not in segment.attributes:
            return self
        grid = self.field.grid
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

        columns = ['DATE', 'WELL', 'MODE', 'CONTROL',
                   'OPT', 'WPT', 'GPT', 'SLPT', 'LPT', 'BHPT']

        wconprod = pd.DataFrame(columns=columns)

        mask_prod = (df['MODE'] != 'INJE').values
        prod = df[mask_prod]
        wconprod['DATE'] = prod['DATE']
        wconprod['WELL'] = segment.name
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

        columns = ['DATE', 'WELL', 'PHASE', 'MODE', 'CONTROL', 'SPIT', 'PIT', 'BHPT']
        wconinje = pd.DataFrame(columns=columns)
        mask_inje = (df['MODE'] == 'INJE').values
        inje = df[mask_inje]
        wconinje['DATE'] = inje['DATE']
        wconinje['WELL'] = segment.name
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

    @apply_to_each_segment
    def calculate_cf(self, segment, beta=1, units='METRIC', cf_aggregation='sum', **kwargs):
        """Calculate connection factor values for each grid block of a segment.

        Parameters
        ----------
        beta : list or ndarray, optional
            Additional multiplicator for productivity index of well. Default 1.
        units : str, optional
            Utints used. Default 'METRIC'.
        cf_aggregation : str, optional
            Aggragation method for connection factors. Default 'sum'.

        Returns
        -------
        comp : Wells
            Wells component with a 'CF' columns in a blocks_info attribute.
        """
        calculate_cf(self.field.rock, self.field.grid, segment,
                     beta, units, cf_aggregation, **kwargs)
        self.set_state(has_cf=True)
        return self

    @apply_to_each_segment
    def apply_perforations(self, segment, current_date=None):
        """Open or close perforation intervals for given time interval.

        Parameters
        ----------
        current_date : pandas.Timestamp, optional
            Final date to open new perforations.

        Returns
        -------
        comp : Wells
            Wells component with an updated blocks_info attribute which contains:
            - projections of welltrack in grid blocks
            - MD values for corresponding grid blocks
            - ratio of perforated part of the well for each grid block.
        """
        segment.apply_perforations(current_date)
        self.set_state(full_perforation=current_date is None)
        return self

    @apply_to_each_segment
    def split_perforations(self, segment):
        """Split well perforations in a way that each perforation correspond to one specific cell."""
        new_perf_df = pd.DataFrame().reindex(
            columns=np.concatenate((segment.perf.columns, ['I', 'J', 'K'])))
        new_perf_df = new_perf_df.astype({
            'I': int,
            'J': int,
            'K': int,
            'COVERED': int,
            'CLOSE': bool
        })
        for _, row in segment.perf.iterrows():
            perforations_new = []
            mdl = row['MDL']
            mdu = row['MDU']
            start = mdl
            index_last = None
            for ((index, _), (_, block)) in zip(enumerate(segment.blocks),
                                                segment.blocks_info.iterrows()):
                if block['MD'] > mdl and block['MD'] < mdu:
                    end = block['MD']
                    i, j, k = segment.blocks[index-1] if index > 0 else (-1, -1, -1)
                    perforations_new.append((start, end, i, j, k))
                    start = block['MD']
                    index_last = index
            end = mdu
            i, j, k = segment.blocks[index_last] if index_last is not None else (-1, -1, -1)
            perforations_new.append((start, end, i, j, k))

            df = pd.DataFrame(perforations_new, columns=['MDL', 'MDU', 'I', 'J', 'K'])
            for column in new_perf_df.columns:
                if column not in df:
                    df[column] = row[column]
            new_perf_df = new_perf_df.append(df, ignore_index=True)

        segment.perf = new_perf_df
        return self

    @apply_to_each_segment
    def update_wpi_mult(self, segment, mults):
        """Update WPI multipliers.

        Parameters
        ----------
        mults : pandas.DataFrame
            Multiplyers to be updated.
            A table with following columns
            -- I:
            -- J:
            -- K: indices
            -- MULT correspnding multipliers

        Returns
        -------
        Wells
            Wells component with updated WPI multipliers in `perf` tables.
        """
        res = pd.merge(segment.perf, mults, on=['I', 'J', 'K'], how='left')
        res['MULT'] = res['MULT_y'].fillna(res['MULT_x'])
        res = res.drop(['MULT_x', 'MULT_y'], axis=1)
        segment.perf = res
        return self

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

    @apply_to_each_segment
    def randomize_events(self, segment, additive=False, clip=None, equality_condition=None, **kwargs):
        """Add random values to events or fill with new random values.

        Parameters
        ----------
        kwargs : dict
            A dict of {keyword: sampler} pairs. A keyword should be one
            of event keywords. A sampler should be callable that accepts parameter size
            (number of samples to generate).
        additive : bool
            If True, add random value to existing one. If False, substitute existing value
            with a new random. Default to False.
        clip : (2, ), tuple
            If not None, apply clip with given min and max values.
        equality_condition: dict, None
            If not None, randomize only those events that satisfy
            df[k] == v for all (k, v) from equality_condition.items()

        Returns
        -------
        wells : Wells
            Wells with randomized events.
        """
        if 'EVENTS' not in segment:
            return self
        df = segment.events
        for k, sampler in kwargs.items():
            if k not in df:
                continue
            eq_cond = pd.Series(np.ones(len(df), dtype=bool))
            if equality_condition is not None:
                for attr, cond in equality_condition.items():
                    eq_cond &= df[attr] == cond
            rnd = sampler(size=eq_cond.sum())
            if additive:
                df.loc[eq_cond, k] += rnd
            else:
                df.loc[eq_cond, k] = rnd
            if clip is not None:
                df.loc[eq_cond, k] = np.clip(df.loc[eq_cond, k], *clip)
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
            arr = segment.welltrack
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=c, **kwargs)
            ax.text(*arr[0, :3], s=segment.name)

        ax.invert_zaxis()
        ax.view_init(azim=60, elev=30)

    def show_rates(self, timesteps=None, wellnames=None, wells2=None, labels=None, figsize=(16, 6)):
        """Plot total or cumulative liquid and gas rates for a chosen node including branches.

        Parameters
        ----------
        timesteps : list of Timestamps
            Dates at which rates were calculated.
        wellnames : array-like
            List of wells to show.
        figsize : tuple
            Figsize for two axes plots.
        wells2 : Wells
            Target model to compare with.
        """
        timesteps = self.result_dates if timesteps is None else timesteps
        wellnames = [node.name for node in PreOrderIter(self.root)]
        return show_rates(self, timesteps=timesteps, wellnames=wellnames, wells2=wells2,
                          labels=labels, figsize=figsize)

    def show_rates2(self, timesteps=None, wellnames=None, wells2=None, labels=None, figsize=(16, 6)):
        """Plot total or cumulative liquid and gas rates for a chosen node including branches
        on two separate axes.

        Parameters
        ----------
        timesteps : list of Timestamps
            Dates at which rates were calculated.
        wellnames : array-like
            List of wells to show.
        figsize : tuple
            Figsize for two axes plots.
        wells2 : Wells
            Target model to compare with.
        """
        timesteps = self.result_dates if timesteps is None else timesteps
        wellnames = [node.name for node in PreOrderIter(self.root)]
        return show_rates2(self, timesteps=timesteps, wellnames=wellnames, wells2=wells2,
                           labels=labels, figsize=figsize)

    def show_blocks_dynamics(self, timesteps=None, wellnames=None, figsize=(16, 6)):
        """Plot liquid or gas rates and pvt props for a chosen block of
        a chosen well segment on two separate axes.

        Parameters
        ----------
        timesteps : list of Timestamps
            Dates at which rates were calculated.
        wellnames : array-like
            List of wells to plot.
        figsize : tuple
            Figsize for two axes plots.
        """
        timesteps = self.result_dates if timesteps is None else timesteps
        wellnames = self.names if wellnames is None else wellnames
        return show_blocks_dynamics(self, timesteps=timesteps, wellnames=wellnames, figsize=figsize)

    def _read_buffer(self, buffer, attr, **kwargs):
        """Load well data from an ASCII file.

        Parameters
        ----------
        buffer : StringIteratorIO
            Buffer to get string from.
        attr : str
            Target keyword.

        Returns
        -------
        comp : Wells
            Wells component with loaded well data.
        """
        if attr == 'WELSPECS':
            return load_welspecs(self, buffer, **kwargs)
        if attr == 'WELSPECL':
            return load_welspecl(self, buffer, **kwargs)
        if attr == 'COMPDAT':
            return load_compdat(self, buffer, **kwargs)
        if attr == 'COMPDATL':
            return load_compdatl(self, buffer, **kwargs)
        if attr == 'COMPDATMD':
            return load_comdatmd(self, buffer, **kwargs)
        if attr == 'WCONPROD':
            return load_wconprod(self, buffer, **kwargs)
        if attr == 'WCONINJE':
            return load_wconinje(self, buffer, **kwargs)
        if attr == 'WEFAC':
            return load_wefac(self, buffer, **kwargs)
        if attr == 'WFRAC':
            return load_wfrac(self, buffer, **kwargs)
        if attr == 'WFRACP':
            return load_wfracp(self, buffer, **kwargs)
        if attr in ["TFIL", "WELLTRACK"]:
            return load_welltracks(self, buffer, **kwargs)
        if attr in ["EFIL", "EFILE", "ETAB"]:
            return load_events(self, buffer, **kwargs)
        if attr in ["HFIL", "HFILE", "HTAB"]:
            return load_history(self, buffer, **kwargs)
        if attr in ["GROU", "GROUP"]:
            return load_group(self, buffer, **kwargs)
        if attr == "GRUPTREE":
            return load_grouptree(self, buffer, **kwargs)
        raise ValueError("Keyword {} is not supported in Wells.".format(attr))

    def _load_rsm(self, *args, **kwargs):
        """Load RSM well data from file."""
        return load_rsm(self, *args, **kwargs)

    def _load_ecl_binary(self, *args, **kwargs):
        """Load results from UNSMRY file."""
        return load_ecl_binary(self, *args, **kwargs)

    @apply_to_each_segment
    def fill_na(self, segment, attr):
        """
        Fill nan values in wells segment attribute.

        Parameters
        ----------
        attr: str
            Attribute name.

        Returns
        -------
        comp : Wells
            Wells with fixed attribute.
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
        comp : Wells
            Wells unchanged.
        """
        with open(path, mode) as f:
            if attr.upper() == 'WELLTRACK':
                for node in self:
                    if 'WELLTRACK' in node and 'COMPDAT' not in node and 'COMPDATL' not in node:
                        f.write('WELLTRACK\t{}\n'.format(node.name))
                        for line in node.welltrack:
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
                raise NotImplementedError("Dump for {} is not implemented.".format(attr.upper()))

    @apply_to_each_segment
    def _get_first_entering_point(self, segment, **kwargs):
        """Calculate grid blocks for the tree of wells.

        Parameters
        ----------
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        comp : Wells
            Wells component with calculated grid blocks and well in block projections.
        """
        _ = kwargs
        grid = self.field.grid.as_corner_point
        if grid._vtk_locator is None or grid._cell_id_d is None: #pylint: disable=protected-access
            grid.create_vtk_locator(use_only_active=True, scaling=False)

        output = find_first_entering_point(segment.welltrack,
                                           grid, grid._vtk_locator, grid._cell_id_d) #pylint: disable=protected-access
        segment._first_entering_point = output #pylint: disable=protected-access

        return self
