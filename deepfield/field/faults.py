"""faults components."""
from itertools import product
import numpy as np
import pandas as pd

from .base_tree import BaseTree
from .faults_load_utils import load_faults, load_multflt
from .faults_dump_utils import write_faults, write_multflt
from .decorators import apply_to_each_segment

FACES = {'X': [1, 3, 5, 7], 'Y': [2, 3, 6, 7], 'Z': [4, 5, 6, 7]}


class Faults(BaseTree):
    """Faults component.

    Contains faults in a single tree structure, faults attributes
    and preprocessing actions.

    Parameters
    ----------
    node : FaultSegment, optional
        Root node for fault's tree.
    """

    def __init__(self, node=None, **kwargs):
        super().__init__(node=node, **kwargs)

    def update(self, data, mode='w', **kwargs):
        """Update tree nodes with new faultsdata. If fault does not exists,
        it will be attached to root.

        Parameters
        ----------
        data : dict
            Keys are fault names, values are dicts with fault attributes.
        mode : str, optional
            If 'w', write new data. If 'a', try to append new data. Default to 'w'.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        out : Faults
            Faults with updated attributes.
        """
        def _get_parent(name):
            if ':' in name:
                return self[':'.join(name.split(':')[:-1])]
            return self.root

        for name in sorted(data):
            fdata = data[name]
            name = name.strip(' \t\'"')
            try:
                segment = self[name]
            except KeyError:
                parent = _get_parent(name)
                segment = self._nodeclass(parent=parent, name=name, ntype='fault')

            for k, v in fdata.items():
                if mode == 'w':
                    setattr(segment, k, v)
                elif mode == 'a':
                    if k in segment.attributes:
                        att = getattr(segment, k)
                        setattr(segment, k, pd.concat([att, v], **kwargs))
                    else:
                        setattr(segment, k, v)
                else:
                    raise ValueError("Unknown mode {}. Expected 'w' (write) or 'a' (append)".format(mode))
        return self

    @apply_to_each_segment
    def get_blocks(self, segment, **kwargs):
        """Calculate grid blocks for the tree of faults.

        Parameters
        ----------
        segment : class instance
            FaultSegment class.
        kwargs : misc
            Any additional named arguments to append.

        Returns
        -------
        comp : faults
            faults component with calculated grid blocks and fault in block projections.
        """
        _ = kwargs
        blocks_fault = []
        xyz_fault = []
        xyz = self.field.grid.xyz #TODO: xyz can be avoided.
        for idx in segment.faults.index:
            cells = segment.faults.loc[idx, ['IX1', 'IX2', 'IY1', 'IY2', 'IZ1', 'IZ2', 'FACE']]
            x_range = range(cells['IX1']-1, cells['IX2'])
            y_range = range(cells['IY1']-1, cells['IY2'])
            z_range = range(cells['IZ1']-1, cells['IZ2'])
            blocks_segment = np.array(list(product(x_range, y_range, z_range)))
            xyz_segment = xyz[blocks_segment[:, 0],
                              blocks_segment[:, 1],
                              blocks_segment[:, 2]][:, FACES[cells['FACE']]]
            blocks_fault.extend(blocks_segment)
            xyz_fault.extend(xyz_segment)

        segment.blocks = np.array(blocks_fault)
        segment.faces_verts = np.array(xyz_fault)
        return self

    def _read_buffer(self, buffer, attr, **kwargs):
        """Load fault data from an ASCII file.

        Parameters
        ----------
        buffer : StringIteratorIO
            Buffer to get string from.
        attr : str
            Target keyword.

        Returns
        -------
        comp : faults
            faults component with loaded fault data.
        """
        if attr == 'FAULTS':
            return load_faults(self, buffer, **kwargs)
        if attr == 'MULTFLT':
            return load_multflt(self, buffer, **kwargs)
        raise ValueError("Keyword {} is not supported in faults.".format(attr))

    def _dump_ascii(self, path, attr, mode='w', **kwargs):
        """Save data into text file.

        Parameters
        ----------
        path : str
            Path to output file.
        attr : str
            Attribute to dump into file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'w'.

        Returns
        -------
        comp : Faults
            Faults unchanged.
        """
        with open(path, mode) as f:
            if attr.upper() == 'FAULTS':
                write_faults(f, self)
            elif attr.upper() == 'MULTFLT':
                write_multflt(f, self)
            else:
                raise NotImplementedError("Dump for {} is not implemented.".format(attr.upper()))
