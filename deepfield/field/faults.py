#pylint: disable=too-many-lines
"""faults and FaultSegment components."""
from copy import deepcopy
from itertools import product
import warnings
import numpy as np
import pandas as pd
import h5py
from anytree import (RenderTree, AsciiStyle, Resolver, PreOrderIter,
                     find_by_attr)

from .fault_segment import FaultSegment
from .base_component import BaseComponent
from .faults_load_utils import load_faults, load_multflt
from .faults_dump_utils import write_faults, write_multflt
from .decorators import apply_to_each_segment

FACES = {'X': [1, 3, 5, 7], 'Y': [2, 3, 6, 7], 'Z': [4, 5, 6, 7]}


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

    Contains faults in a single tree structure, faults attributes
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
        self.init_state(has_blocks=False)

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
    def names(self):
        """List of fault names."""
        return [node.name for node in self]

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
        """Update tree nodes with new faultsdata. If fault does not exists,
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
        out : Faults
            Faults with updated attributes.
        """
        def _get_parent(name):
            if ':' in name:
                return self[':'.join(name.split(':')[:-1])]
            return self.root

        for name in sorted(faultsdata):
            data = faultsdata[name]
            name = name.strip(' \t\'"')
            try:
                segment = self[name]
            except KeyError:
                parent = _get_parent(name)
                segment = FaultSegment(parent=parent, name=name, ntype='fault')

            for k, v in data.items():
                if mode == 'w':
                    setattr(segment, k, v)
                elif mode == 'a':
                    if k in segment.attributes:
                        att = getattr(segment, k)
                        setattr(segment, k, pd.concat([att, v], **kwargs))
                    else:
                        setattr(segment, k, v)
                else:
                    raise ValueError("Unknown mode {}. Expected 'w' (write) or 'a' (append)".format(mode))
        return self

    def drop(self, names):#pylint:disable=arguments-renamed
        """Detach faults by names.

        Parameters
        ----------
        names : str, array-like
            Faults to be detached.

        Returns
        -------
        out : Faults
            Faults without detached segments.
        """
        for name in np.atleast_1d(names):
            try:
                self[name].parent = None
            except KeyError:
                continue
        return self

    @property
    def resolver(self):
        """Tree resolver."""
        return self._resolver

    def glob(self, name):
        """Return instances at ``name`` supporting wildcards."""
        return self.resolver.glob(self.root, name)

    def render_tree(self):
        """Print tree structure."""
        print(RenderTree(self.root, style=AsciiStyle()).by_attr())
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
            x_range = range(cells['IX1']-1, cells['IX2'])
            y_range = range(cells['IY1']-1, cells['IY2'])
            z_range = range(cells['IZ1']-1, cells['IZ2'])
            blocks_segment = np.array(list(product(x_range, y_range, z_range)))
            xyz_segment = self.field.grid.xyz[blocks_segment[:, 0],
                                              blocks_segment[:, 1],
                                              blocks_segment[:, 2]][:, FACES[cells['FACE']]]
            blocks_fault.extend(blocks_segment)
            xyz_fault.extend(xyz_segment)

        segment.blocks = np.array(blocks_fault)
        segment.faces_verts = np.array(xyz_fault)

        self.set_state(has_blocks=True)
        return self

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
        comp : Faults
            Faults unchanged.
        """
        with open(path, mode) as f:
            if attr.upper() == 'FAULTS':
                write_faults(f, self)
            elif attr.upper() == 'MULTFLT':
                write_multflt(f, self)
            else:
                raise NotImplementedError("Dump for {} is not implemented.".format(attr.upper()))

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
        comp : Faults
            Faults unchanged.
        """
        _ = kwargs
        with h5py.File(path, mode) as f:
            faults = f[self.class_name] if self.class_name in f else f.create_group(self.class_name)
            if state:
                for k, v in self.state.as_dict().items():
                    faults.attrs[k] = v
            for fault in PreOrderIter(self.root):
                if fault.is_root:
                    continue
                fault_path = fault.fullname
                if fault.name == 'data':
                    raise ValueError("Name 'data' is not allowed for nodes.")
                grp = faults[fault_path] if fault_path in faults else faults.create_group(fault_path)
                grp.attrs['ntype'] = fault.ntype
                if 'data' not in grp:
                    grp_faults_data = faults.create_group(fault_path + '/data')
                else:
                    grp_faults_data = grp['data']
                for att, data in fault.items():
                    if isinstance(data, pd.DataFrame):
                        continue
                    if att in grp_faults_data:
                        del grp_faults_data[att]
                    grp_faults_data.create_dataset(att, data=data)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for fault in PreOrderIter(self.root):
                if fault.is_root:
                    continue
                for att, data in fault.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_hdf(path, key='/'.join([self.class_name, fault.fullname,
                                                            'data', att]), mode='a')

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

        def update_faults(grp, parent=None):
            """Build tree recursively following HDF5 node hierarchy."""
            if parent is None:
                fault = self.root
            else:
                ntype = grp.attrs.get('ntype', None)
                if ntype is None: #backward compatibility, will be removed in a future
                    ntype = 'group' if grp.attrs.get('is_group', False) else 'fault'
                fault = FaultSegment(parent=parent, name=grp.name.split('/')[-1], ntype=ntype)
            for k, v in grp.items():
                if k == 'data':
                    for att in v.keys() if attrs is None else attrs:
                        try:
                            data = v[att]
                        except KeyError:
                            continue
                        if isinstance(data, h5py.Group):
                            data = pd.read_hdf(path, key='/'.join([grp.name, 'data', att]), mode='r')
                            setattr(fault, att, data)
                        else:
                            setattr(fault, att, data[()])
                else:
                    update_faults(v, fault)

        with h5py.File(path, 'r') as f:
            self.set_state(**dict(f[self.class_name].attrs.items()))
            update_faults(f[self.class_name])
        return self
