"""BaseTree components."""
from copy import deepcopy
from typing import Generic, Self, TypeVar
import warnings
from weakref import ref
import numpy as np
import pandas as pd
import h5py
from anytree import RenderTree, AsciiStyle, Resolver, PreOrderIter, PostOrderIter, find_by_attr

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
    
    def __iter__(self) -> Self:
        return self

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

    def glob(self, name):
        """Return instances at ``name`` supporting wildcards."""
        return self.resolver.glob(self.root, name)

    def tree(self):
        """Print tree structure."""
        print(RenderTree(self.root, style=AsciiStyle()).by_attr())
        return self
