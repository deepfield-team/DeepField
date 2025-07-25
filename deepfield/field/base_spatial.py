"""SpatialComponent class."""
from copy import deepcopy
import numpy as np
import skimage
from skimage.transform import rescale, resize
import scipy

from .base_component import BaseComponent
from .decorators import apply_to_each_input, add_actions, extract_actions, TEMPLATE_DOCSTRING

ACTIONS_DICT = {
    "pad": (np.pad, "numpy.pad", "padded array"),
    "flip": (np.flip, "numpy.flip", "reversed order of elements in an array along the given axis"),
    "clip": (np.clip, "numpy.clip", "array of cliped values"),
    "rot90": (np.rot90, "numpy.rot90", "rotated an array by 90 degrees in the plane specified by axes"),
    "gradient": (np.gradient, "numpy.gradient", "gradient"),
    "resize": (resize, "skimage.transform.resize", "resize"),
    "rescale": (rescale, "skimage.transform.rescale", "rescale"),
    "crop": (skimage.util.crop, "crop", "cropped array by crop_width along each dimension"),
    "random_noise": (skimage.util.random_noise, "random_noise",
                     "array with added random noise of various types"),
}

@add_actions(extract_actions(scipy.ndimage), TEMPLATE_DOCSTRING)
@add_actions(ACTIONS_DICT, TEMPLATE_DOCSTRING)
class SpatialComponent(BaseComponent):
    """Base component for spatial-type attributes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_state(spatial=None)

    def sample_crops(self, attr, shape, size=1):
        """Sample random crops of fixed shape.

        Parameters
        ----------
        attr : str, array-like
            Attributes to sample crops from. If None, use all attributes.
        shape : tuple
            Shape of crops.
        size : int, optional
            Number of crops to sample. Default to 1.

        Returns
        -------
        crops : ndarray
            Sampled crops.
        """
        is_list = True
        if isinstance(attr, str):
            attr = [attr]
            is_list = False
        if attr is None:
            attr = self.attributes
        data_shape = np.array(getattr(self, attr[0]).shape)
        valid_range = data_shape - np.array(shape)
        before = np.array([np.random.randint(0, d, size=size) for d in valid_range]).T
        after = valid_range - before
        res = [self.crop(attr=attr, crop_width=list(zip(before[i], after[i]))) for i in range(size)]
        res = np.swapaxes(res, 0, 1)
        return res[0] if is_list else res

    def ravel(self, attr, **kwargs):
        """Returns ravel representation for attributes with pre-defined ravel transformation.

        Parameters
        ----------
        attr : str, array of str
            Attribute to ravel.
        kwargs : misc
            Additional named arguments.

        Returns
        -------
        out : raveled attribute.
        """
        return self._ravel(attr=attr, **kwargs)

    def _ravel(self, attr, **kwargs):
        """Ravel transformations."""
        return super().ravel(attr=attr, **kwargs)

    def to_spatial(self, attr=None, **kwargs):
        """Bring component to spatial state.

        Parameters
        ----------
        attr : str, array of str
            Attribute to ravel.
        inplace : bool
            Modify сomponent inplace.
        kwargs : misc
            Additional named arguments.

        Returns
        -------
        out : component with spatial attributes.
        """
        self._to_spatial(attr=attr, **kwargs)
        return self

    @apply_to_each_input
    def _to_spatial(self, attr, **kwargs):
        """Spatial transformations."""
        _ = self, attr, kwargs
        raise NotImplementedError()

    def _make_data_dump(self, attr, fmt=None, **kwargs):
        _ = fmt, kwargs
        return self.ravel(attr=attr, order='F')

    def load(self, path_or_buffer, **kwargs):
        super().load(path_or_buffer, **kwargs)
        self.to_spatial()

    def copy_attribute(self, attr1, attr2, box=None):
        """Copy attribute values to another atribute.

        Parameters
        ----------
        attr1 : str
            Attribute to be copied.
        attr2 : str
            Destination attribute
        box : Sequence[int, int, int, int, int, int] or None, optional
            (i1, i2, j1, j2, k1, k2) - Box in which
            attribute values should be copied (`attr2[i1:i2, j1:j2, k1:k2]) = attr1[i1:i2, j1:j2, k1:k2]`),
            by default None.

        Returns
        -------
        self.__class__
            self
        """
        if box is not None:
            getattr(self, attr2)[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = getattr(
                self, attr1)[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        setattr(self, attr2, deepcopy(getattr(self, attr1)))
        return self

    def multiply_attribute(self, attr, multiplier, box):
        """Multiply attribute by a constant.

        Parameters
        ----------
        attr : str
            Attribute to be modfied.
        multiplier : float
            Multiplier.
        box : Sequence[int, int, int, int, int, int]
            (i1, i2, j1, j2, k1, k2) - Box in which
            attribute values should be multiplyed by a constant
            (`attr[i1:i2, j1:j2, k1:k2]) = attr[i1:i2, j1:j2, k1:k2]*multiplier`).

        Returns
        -------
        self.__class__
            self
        """
        getattr(self, attr)[box[0]:box[1], box[2]:box[3], box[4]:box[5]] *= multiplier
        return self

    def equals_attribute(self, attr, val, box=None, dtype=None, create=False):
        """Set attribute values to a constant.

        Parameters
        ----------
        attr : str
            Attribute to be modified.
        multiplier : float
            Multiplier.
        box : Sequence[int, int, int, int, int, int]
            (i1, i2, j1, j2, k1, k2) - Box in which
            attribute values should be set to a constant
            (`attr[i1:i2, j1:j2, k1:k2]) = val`), by default None.
        dtype: type
            Type of created array.
        create: bool
            If `create==True` attribute is created, in case it does not exist.

        Returns
        -------
        self.__class__
            self
        """
        dimens = self.field.grid.dimens
        if attr not in self.attributes and create:
            dtype = float if dtype is None else dtype
            setattr(self, attr, np.zeros(dimens, dtype=dtype))
        if box is None:
            box = (0, dimens[0], 0, dimens[1], 0, dimens[2])
        getattr(self, attr)[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = val

    def add_to_attribute(self, attr, addition, box):
        """Add a constant to an attribute.

        Parameters
        ----------
        attr : str
            Attribute to be modfied.
        addition : float
            Addition.
        box : Sequence[int, int, int, int, int, int]
            (i1, i2, j1, j2, k1, k2) - Box in which
            a constant should be added to an attribute
            (`attr[i1:i2, j1:j2, k1:k2]) = attr[i1:i2, j1:j2, k1:k2] + addition`).

        Returns
        -------
        self.__class__
            self
        """
        getattr(self, attr)[box[0]:box[1], box[2]:box[3], box[4]:box[5]] += addition
        return self

    @apply_to_each_input
    def pad_na(self, attr, fill_na=0., inplace=True):
        """Add dummy cells into the state vector in the positions of non-active cells if necessary.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be padded with non-active cells.
        actnum: array-like of type bool
            Vector representing a mask of active and non-active cells.
        fill_na: float
            Value to be used as filler.
        inplace: bool
            Modify сomponent inplace.

        Returns
        -------
        output : component if inplace else padded attribute.
        """
        raise NotImplementedError
