"""TreeSegment components."""
from anytree import NodeMixin

from .base_component import BaseComponent


class BaseTreeNode(BaseComponent, NodeMixin):
    """Well's node.

    Parameters
    ----------
    name : str, optional
        Node's name.
    is_group : bool, optional
        Should a node represet a group of nodes. Default to False.

    Attributes
    ----------
    is_group : bool
        Indicator of a group.
    name : str
        Node's name.
    fullname : str
        Node's full name from root.
    """

    def __init__(self, *args, parent=None, children=None, name=None, ntype=None, **kwargs):
        super().__init__(*args, **kwargs)
        super().__setattr__('parent', parent)
        self._name = name
        self._ntype = ntype
        if children is not None:
            super().__setattr__('children', children)

    def copy(self):
        """Returns a deepcopy. Cached properties are not copied."""
        copy = super().copy()
        copy._name = self._name #pylint: disable=protected-access
        copy._ntype = self._ntype #pylint: disable=protected-access
        return copy

    @property
    def is_group(self):
        """Check that node is a group of wells."""
        return self._ntype == 'group'

    @property
    def ntype(self):
        """Node's type."""
        return self._ntype

    @property
    def name(self):
        """Node's name."""
        return self._name

    @property
    def fullname(self):
        """Full name from root."""
        return self.separator.join([node.name for node in self.path[1:]])
