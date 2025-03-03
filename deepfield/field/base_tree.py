"""BaseTree components."""
from copy import deepcopy
import warnings
from weakref import ref
import numpy as np
import pandas as pd
import h5py
from anytree import (RenderTree, AsciiStyle, Resolver, PreOrderIter, PostOrderIter,
                     find_by_attr)

from .base_tree_node import BaseTreeNode
from .base_component import BaseComponent


class IterableTree:
    """Tree iterator."""
    def __init__(self, root):
        self.iter = PreOrderIter(root)

    def __next__(self):
        x = next(self.iter)
        if x.ntype == 'group':
            return next(self)
        return x


class BaseTree(BaseComponent):
    """Base tree component.

    Contains nodes and groups in a single tree structure.

    Parameters
    ----------
    node : TreeSegment, optional
        Root node for the tree.
    """

    def __init__(self, node=None, nodeclass=None, **kwargs):
        super().__init__(**kwargs)
        nodeclass = BaseTreeNode if nodeclass is None else nodeclass
        self._root = nodeclass(name='FIELD', ntype="group",
                               field=self._field) if node is None else node
        self._resolver = Resolver()
        self._nodeclass = nodeclass

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
    def field(self):
        return self._field()

    @field.setter
    def field(self, field):
        """Set field to which component belongs."""
        if isinstance(field, ref) or field is None:
            self._field = field
            return self
        self._field = ref(field)
        if hasattr(self, 'root'):
            for node in self:
                node.field = field
        return self

    @property
    def root(self):
        """Tree root."""
        return self._root

    @property
    def resolver(self):
        """Tree resolver."""
        return self._resolver

    @property
    def names(self):
        """List of well names."""
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
        return IterableTree(self.root)

    def __contains__(self, key):
        return find_by_attr(self.root, key) is not None

    def update(self, data, mode='w', **kwargs):
        """Update tree nodes with new data. If node does not exists,
        it will be attached to the root.

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
        out : self
            Tree with updated attributes.
        """
        _ = self, data, mode, kwargs
        raise NotImplementedError()

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
        out : self
            Component with a new group added.
        """
        group = self._nodeclass(name=group_name, parent=self.root, ntype="group")
        nodes = [self[name] for name in node_names]
        node_types = [node.ntype for node in nodes]
        if np.unique(node_types).size > 1:
            raise ValueError("Can not group mixture of nodes and groups in a new group.")
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
        out : self
            Component without detached nodes.
        """
        for name in np.atleast_1d(names):
            try:
                self[name].parent = None
            except KeyError:
                continue
        return self

    def drop_empty_groups(self, logger=None):
        """Drop groups without nodes in descendants."""
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
        comp : self
            A component with loaded attributes.
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
                    ntype = 'group' if grp.attrs.get('is_group', False) else 'well'
                node = self._nodeclass(parent=parent, name=grp.name.split('/')[-1],
                                       ntype=ntype, field=self.field)
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
        comp : self
            Unchanged component.
        """
        _ = kwargs
        with h5py.File(path, mode) as f:
            nodes = f[self.class_name] if self.class_name in f else f.create_group(self.class_name)
            if state:
                for k, v in self.state.as_dict().items():
                    nodes.attrs[k] = v
            for node in PreOrderIter(self.root):
                if node.is_root:
                    continue
                node_path = node.fullname
                if node.name == 'data':
                    raise ValueError("Name 'data' is not allowed for nodes.")
                grp = nodes[node_path] if node_path in nodes else nodes.create_group(node_path)
                grp.attrs['ntype'] = node.ntype
                if 'data' not in grp:
                    grp_data = nodes.create_group(node_path + '/data')
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
