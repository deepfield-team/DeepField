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

FACES = {'X': [1, 3, 5, 7], 'Y': [0, 1, 4, 5], 'Z': [4, 5, 6, 7]}

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
                        spatial=True)

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
        def _get_parent(name):
            if ':' in name:
                return self[':'.join(name.split(':')[:-1])]
            return self.root

        for name in sorted(faultsdata):
            data = faultsdata[name]
            name = name.strip(' \t\'"')
            try:
                node = self[name]
            except KeyError:
                parent = _get_parent(name)
                node = FaultSegment(parent=parent, name=name, ntype='fault')

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

    def glob(self, name):
        """Return instances at ``name`` supporting wildcards."""
        return self.resolver.glob(self.root, name)

    def render_tree(self):
        """Print tree structure."""
        print(RenderTree(self.root, style=AsciiStyle()).by_attr())
        return self

    def blocks_ravel(self):
        """Transforms block coordinates into 1D representation."""
        if not self.state.spatial:
            return self
        self.set_state(spatial=False)
        return self._blocks_ravel()

    def blocks_to_spatial(self):
        """Transforms block coordinates into 3D representation."""
        if self.state.spatial:
            return self
        self.set_state(spatial=True)
        return self._blocks_to_spatial()

    @apply_to_each_segment
    def _blocks_ravel(self, segment):
        grid = self._field().grid
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

    @apply_to_each_segment
    def _blocks_to_spatial(self, segment):
        grid = self._field().grid
        if 'BLOCKS' not in segment or not len(segment.blocks):
            return self
        res = active_ind_to_full_ind(segment.blocks, grid)
        res = np.unravel_index(res, shape=grid.dimens, order='F')
        res = np.stack(res, axis=1)
        setattr(segment, 'BLOCKS', res)
        return self

    @apply_to_each_segment
    def get_blocks(self, segment, **kwargs):
        """Calculate grid blocks for the tree of faults.

        Parameters
        ----------
        segment : class instance
            FaultSegment class.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        comp : faults
            faults component with calculated grid blocks and fault in block projections.
        """
        _ = kwargs
        blocks_fault = []
        xyz_fault = []
        for idx in segment.faults.index:
            cells = segment.faults.loc[idx, ['IX1', 'IX2', 'IY1', 'IY2', 'IZ1', 'IZ2', 'FACE']]
            X_range = range(cells['IX1']-1, cells['IX2'])
            Y_range = range(cells['IY1']-1, cells['IY2'])
            Z_range = range(cells['IZ1']-1, cells['IZ2'])
            blocks_segment = np.array(list(product(X_range, Y_range, Z_range)))
            xyz_segment = self._field().grid.xyz[blocks_segment[:, 0], 
                                                 blocks_segment[:, 1], 
                                                 blocks_segment[:, 2]][:, FACES[cells['FACE']]]
            blocks_fault.extend(blocks_segment)
            xyz_fault.extend(xyz_segment)
        
        segment.blocks = np.array(blocks_fault)
        segment.blocks_xyz = np.array(xyz_fault)

        self.set_state(has_blocks=True)
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
