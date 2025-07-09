"""Wells components."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anytree import PreOrderIter, PostOrderIter

from .parse_utils.ascii import INT_NAN
from .well_segment import WellSegment
from .base_tree import BaseTree
from .rates import show_rates, show_blocks_dynamics
from .grids import OrthogonalGrid
from .getting_wellblocks import get_wellblocks_vtk, get_wellblocks_compdat
from .wells_dump_utils import write_perf, write_events, write_schedule, write_welspecs
from .wells_load_utils import (load_rsm, load_ecl_binary, load_group, load_grouptree,
                               load_welspecs, load_welspecl, load_compdat, load_compdatl,
                               load_comdatmd, load_wconprod, load_wconinje, load_welltracks,
                               load_events, load_history, load_wefac, load_wfrac, load_wfracp,
                               DEFAULTS, VALUE_CONTROL)
from .decorators import apply_to_each_segment


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
        centroids = grid.xyz.mean(axis=-2) #TODO: xyz can be avoided.
        for _ in range(len(df)):
            dist = np.linalg.norm(df[['I', 'J', 'K1']] - root, axis=1)
            row = df.iloc[[dist.argmin()]]
            track.append(centroids[int(row.iloc[0]['I'])-1,
                                   int(row.iloc[0]['J'])-1,
                                   int(row.iloc[0]['K1'])-1])
            track.append(centroids[int(row.iloc[0]['I'])-1,
                                   int(row.iloc[0]['J'])-1,
                                   int(row.iloc[0]['K2'])-1])
            root = row[['I', 'J', 'K2']].values.astype(float).ravel()
            df = df.drop(row.index)
        track = pd.DataFrame(track).drop_duplicates().values
        segment.WELLTRACK = np.concatenate([track, np.full((len(track), 1), np.nan)], axis=1)
        return self

    @apply_to_each_segment
    def get_blocks(self, segment, logger=None):
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

        if 'COMPDAT' in segment.attributes or 'COMPDATL' in segment.attributes:
            if 'COMPDAT' in segment.attributes:
                compdat = segment.compdat
            elif (segment.compdatl['LGR']=='GLOBAL').all():
                compdat = segment.compdatl
            else:
                logger.warning('Well {}: can not get blocks from COMPDATL data.'.format(segment.name))

            segment.blocks = get_wellblocks_compdat(compdat)
            if isinstance(self.field.grid, OrthogonalGrid):
                h_well = np.stack([(0, 0, self.field.grid.dz[i[0], i[1], i[2]])
                                   for i in segment.blocks])
            else:
                h_well = np.full(segment.blocks.shape, np.NaN)
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])

        else:
            blocks, points, mds = get_wellblocks_vtk(segment.welltrack, grid)

            segment.blocks = blocks
            h_well = abs(points[:, 1] - points[:, 0])
            segment.blocks_info = pd.DataFrame(h_well, columns=['Hx', 'Hy', 'Hz'])
            segment.blocks_info['MDU'] = mds[:, 0]
            segment.blocks_info['MDL'] = mds[:, 1]
            segment.blocks_info['Enter_point'] = list(points[:, 0])
            segment.blocks_info['Leave_point'] = list(points[:, 1])

        segment.blocks_info = segment.blocks_info.assign(
            PERF_RATIO=None if len(segment.blocks_info) == 0 else 0,
            RAD=None if len(segment.blocks_info) == 0 else DEFAULTS['RAD'],
            SKIN=None if len(segment.blocks_info) == 0 else DEFAULTS['SKIN'],
            MULT=None if len(segment.blocks_info) == 0 else DEFAULTS['MULT'],
        )
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
