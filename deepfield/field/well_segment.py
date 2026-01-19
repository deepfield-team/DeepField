#pylint: disable=too-many-lines
"""Wells and WellSegment components."""
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from .base_tree_node import BaseTreeNode, NodeAttributeView


class WellSegment(BaseTreeNode):
    """Well's node.

    Parameters
    ----------
    name : str, optional
        Node's name.
    is_group : bool, optional
        Should a node represet a group of wells. Default to False.

    Attributes
    ----------
    is_group : bool
        Indicator of a group.
    is_main_branch : bool
        Indicator of a main branch.
    name : str
        Node's name.
    fullname : str
        Node's full name from root.
    """

    wconprod: NodeAttributeView = NodeAttributeView('WCONPROD', 'WELL')
    wconinj: NodeAttributeView = NodeAttributeView('WCONINJ', 'WELL')
    welspecs: NodeAttributeView = NodeAttributeView('WELSPECS', 'WELL')
    welspecsl: NodeAttributeView = NodeAttributeView('WELSPECSL', 'WELL')
    compdat: NodeAttributeView = NodeAttributeView('COMPDAT', 'WELL')
    compdatl: NodeAttributeView = NodeAttributeView('COMPDATL', 'WELL')
    compdatmd: NodeAttributeView = NodeAttributeView('COMPDATMD', 'WELL')
    wefac: NodeAttributeView = NodeAttributeView('WEFAC', 'WELL')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._blocks: NDArray[np.integer] | None = None

    @property
    def is_main_branch(self):
        """Check that node in a main well's branch."""
        return (not self._ntype == 'group') and (':' not in self.name)

    @property
    def total_rates(self):
        """Total rates for the current node and all its branches."""
        columns = ['DATE', 'WOPR', 'WWPR', 'WGPR', 'WFGPR']
        if 'RESULTS' not in self:
            df = pd.DataFrame(columns=columns).set_index('DATE')
        else:
            df = self.results[[x for x in columns if x in self.results]].set_index('DATE')
        for node in self.children:
            df = df.add(node.total_rates.set_index('DATE'), fill_value=0)
        return df.reset_index()

    @property
    def cum_rates(self):
        """Cumulative rates for the current node and all its branches."""
        return self.total_rates.set_index('DATE').cumsum().reset_index()

    @property
    def blocks(self):
        return self._blocks
    
    @blocks.setter
    def blocks(self, val: NDArray[np.integer]):
        self._blocks = val
