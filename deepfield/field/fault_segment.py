"""FaultSegment component."""
#pylint: disable=too-many-lines
from anytree import NodeMixin
from .base_component import BaseComponent


class FaultSegment(BaseComponent, NodeMixin):
    """Fault's node.

    Parameters
    ----------
    name : str, optional
        Node's name.

    Attributes
    ----------
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
