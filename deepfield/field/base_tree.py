"""BaseTree components."""
from typing import Self
from weakref import ref
from anytree import RenderTree, AsciiStyle, Resolver, PreOrderIter, find_by_attr

from .base_tree_node import BaseTreeNode
from .base_component import BaseComponent


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
