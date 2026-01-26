#pylint: disable=too-many-lines
"""Wells and WellSegment components."""
from typing import override
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from .base_tree_node import BaseTreeNode, NodeAttributeViewDataFrame, NodeAttributeViewDict


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

    wconprod: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('WCONPROD', 'WELL')
    wconinj: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('WCONINJ', 'WELL')
    welspecs: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('WELSPECS', 'WELL')
    welspecsl: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('WELSPECSL', 'WELL')
    compdat: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('COMPDAT', 'WELL')
    compdatl: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('COMPDATL', 'WELL')
    compdatmd: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('COMPDATMD', 'WELL')
    wefac: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('WEFAC', 'WELL')
    results: NodeAttributeViewDataFrame = NodeAttributeViewDataFrame('RESULTS', 'WELL')
    welltrack: NodeAttributeViewDict = NodeAttributeViewDict('WELLTRACK', None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._blocks: NDArray[np.int_] | None = None
        self._blocks_info: pd.DataFrame | None = None

    @property
    @override
    def root_component(self):
        return self.field.wells

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
    def blocks(self, val: NDArray[np.int_]):
        self._blocks = val
    @property
    def blocks_info(self)-> pd.DataFrame | None:
        """The blocks_info property."""
        return self._blocks_info

    @blocks_info.setter
    def blocks_info(self, value: pd.DataFrame | None):
        self._blocks_info = value
