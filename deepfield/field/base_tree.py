"""Base tree component."""
from typing import Self
from weakref import ref
import pandas as pd
from anytree import Node, RenderTree, AsciiStyle, Resolver, PreOrderIter, find_by_attr

from .base_component import BaseComponent

class BaseTreeNode(Node):
    """Tree's node.

    Parameters
    ----------
    name : str
        Name of the node.
    key : str, optional
        Key used to query node data.
    is_group : bool, optional
        Should a node represet a group of nodes. Default False.
    """

    def __init__(self, name, key=None, is_group=False, **kwargs):
        super().__init__(name, **kwargs)
        self._key = key
        self._is_group = is_group

    def __getattr__(self, attr) -> pd.DataFrame | None:
        if attr.startswith('_'):
            raise AttributeError(attr)
        data = getattr(self.root.component(), attr)
        if self.is_root:
            return data
        return None if data is None else data[data[self.key] == self.name]

    def __contains__(self, x: str):
        try:
            return getattr(self, x) is not None
        except AttributeError:
            return False

    @property
    def is_group(self):
        """Check that node is a group of nodes."""
        return self._is_group

    @property
    def key(self):
        """Node's type."""
        return self._key

    @property
    def fullname(self):
        """Full name from root."""
        return self.separator.join([node.name for node in self.path[1:]])


class IterableTree:
    """Tree iterator excluding group nodes."""
    def __init__(self, root):
        self.iter = PreOrderIter(root)

    def __next__(self):
        x = next(self.iter)
        if x.is_group:
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

    def __init__(self, root=None, **kwargs):
        super().__init__(**kwargs)
        self._root = root if root is not None else BaseTreeNode(name='root')
        self._root.component = ref(self)
        self._resolver = Resolver()

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
        """List of node names excluding group nodes."""
        return [node.name for node in self]

    def __getitem__(self, key):
        node = find_by_attr(self.root, key)
        if node is None:
            raise KeyError(key)
        return node

    def __iter__(self):
        return IterableTree(self.root)

    def glob(self, name):
        """Return instances at ``name`` supporting wildcards."""
        return self.resolver.glob(self.root, name)

    def render_tree(self):
        """Print tree structure."""
        print(RenderTree(self.root, style=AsciiStyle()).by_attr())
        return self

    def build_tree(self):
        """Build tree from component's data."""
        raise NotImplementedError()
