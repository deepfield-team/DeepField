"""Faults component."""
from itertools import product
import numpy as np

from .base_component import Attribute
from .base_tree import BaseTree, BaseTreeNode

from .utils.decorators import apply_to_each_node

FACES = {'X': [1, 3, 5, 7], 'Y': [2, 3, 6, 7], 'Z': [4, 5, 6, 7]}

class FaultsNode(BaseTreeNode):
    """Faults node."""

FAULTS_ATTRIBUTES = ['FAULTS', 'MULTFLT']

class Faults(BaseTree):
    """Faults component."""
    _attributes_to_load: list[Attribute] = [
        Attribute(attr, 'GRID', attr) for attr in FAULTS_ATTRIBUTES]

    def __init__(self, **kwargs):
        root = FaultsNode(name='FIELD', is_group=True)
        super().__init__(root=root, **kwargs)

    def build_tree(self):
        """Build tree from component's data."""
        if 'FAULTS' in self:
            faults = self.faults
        else:
            return self

        for name in faults.NAME.unique():
            FaultsNode(parent=self.root, name=name, key='NAME')

        return self

    @apply_to_each_node
    def get_blocks(self, segment, **kwargs):
        """Calculate grid blocks for the tree of faults."""
        _ = kwargs
        blocks_fault = []
        xyz_fault = []
        grid = self.field.grid
        for _, cells in segment.faults.iterrows():
            x_range = range(cells['I1']-1, cells['I2'])
            y_range = range(cells['J1']-1, cells['J2'])
            z_range = range(cells['K1']-1, cells['K2'])
            blocks_segment = np.array(list(product(x_range, y_range, z_range)))
            xyz_segment = grid.get_xyz(blocks_segment)[:, FACES[cells['FACE']]]
            blocks_fault.extend(blocks_segment)
            xyz_fault.extend(xyz_segment)

        segment.blocks = np.array(blocks_fault)
        segment.faces_verts = np.array(xyz_fault)
        return self
