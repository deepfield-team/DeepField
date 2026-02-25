"""BaseTreeNode class."""
from anytree import Node
import pandas as pd


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
