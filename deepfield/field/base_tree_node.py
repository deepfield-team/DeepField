"""TreeSegment components."""
from __future__ import annotations
from typing import TypeVar
from anytree import NodeMixin
import pandas as pd

from .base_component import BaseComponent

T = TypeVar('T', dict[str, pd.DataFrame], pd.DataFrame)

class NodeAttributeView():
    def __init__(self, att: str, key: str | None) -> None:
        self._att: str = att
        self._key: str | None = key

    def __get__(self, obj: BaseTreeNode, objtype=None) -> pd.DataFrame | None:
        _ = objtype
        name = obj.name
        assert isinstance(name, str)
        component = obj.root_component
        comp_att = getattr(component, self._att)
        if comp_att is None:
            return None
        return self._get(comp_att, name)

    def _get(self, att: pd.DataFrame, name: str):
        assert self._key is not None
        return att[att[self._key] == name]

    def __set__(self, obj: BaseTreeNode, value: pd.DataFrame) -> None:
        raise NotImplementedError()


class BaseTreeNode(BaseComponent, NodeMixin):
    """Tree's node.

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

    @property
    def root_component(self) -> BaseComponent | None:
        return None

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
