"""TreeSegment components."""
from __future__ import annotations
from weakref import ref
from anytree import NodeMixin
import pandas as pd

from .base_component import BaseComponent

class NodeAttributeView:
    def __init__(self, att: str, key: str) -> None:
        self._att: str = att
        self._key: str = key
    def __get__(self, obj: BaseTreeNode, objtype=None):
        _ = objtype
        name = obj.name
        component = obj.root_component
        comp_att = getattr(component, self._att)
        if comp_att is None:
            return comp_att
        assert isinstance(comp_att, pd.DataFrame)
        return comp_att[comp_att[self._key] == name]

    def __set__(self, obj: BaseTreeNode, value: pd.DataFrame):
        raise NotImplementedError()

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

    def __init__(self, root_component: BaseComponent, *args, parent=None, children=None, name=None, ntype=None, **kwargs):
        super().__init__(*args, **kwargs)
        super().__setattr__('parent', parent)
        self._name = name
        self._ntype = ntype
        self._root_component: ref[BaseComponent] | None = None
        self.root_component = root_component
        if children is not None:
            super().__setattr__('children', children)

    @property
    def root_component(self) -> BaseComponent:
        assert self._root_component is not None
        res = self._root_component()
        assert res is not None
        return res

    @root_component.setter
    def root_component(self, val: BaseComponent):
        self._root_component = ref(val)
        return self

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
