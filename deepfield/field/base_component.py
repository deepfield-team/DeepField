"""Base compoment."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Generic, Self, TypeVar, Sequence, TypeAlias, override
from copy import deepcopy
import logging
import warnings
from weakref import ref
import numpy as np

import resdp
import resdp.binary

from .utils.decorators import apply_to_each_input


if TYPE_CHECKING:
    from .field import Field


AttributeLoaderType: TypeAlias = Callable[
    [resdp.DataType, resdp.binary.BinaryData, logging.Logger], resdp.ValueType]


class BaseComponent:
    """Base class for reservoir model components."""
    _attributes_to_load: list[Attribute[Self]] = []
    def __init__(self, data=None, field=None):
        self._field = None
        self._attributes: list[Attribute] = []
        self._binary_attributes = []
        if data is not None:
            self._attributes = data['attributes']
            for att in self._attributes:
                att.component = self
            self._binary_attributes = data['binary_attributes']
        if field is not None:
            self.field = field

    @property
    def field(self) -> Field:
        """Field associated with the component."""
        return self._field()

    @field.setter
    def field(self, field):
        """Set field to which component belongs."""
        self._field = field if isinstance(field, ref) or field is None else ref(field)
        return self

    @property
    def attributes(self) -> Sequence[str]:
        """Names of attributes."""
        return tuple((attr.name for attr in self._attributes if attr.value is not None))

    @property
    def binary_attributes(self) -> Sequence[str]:
        """Names of binary attributes."""
        return self._binary_attributes

    @property
    def empty(self):
        """True if component is empty else False."""
        return not self._attributes

    def items(self):
        """Returns pairs of attribute's names and data."""
        return ((attr.name, attr.value) for attr in self._attributes if attr.value is not None)

    def __getattr__(self, key):
        for attr in self._attributes:
            if key.upper() == attr.name:
                return attr.value
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")

    def data_dict(self):
        """Create dict from attributes."""
        return {'attributes': deepcopy(self._attributes), 'binary_attributes': self.binary_attributes.copy()}

    def __setattr__(self, key, value):
        if (key[0] == '_') or (key in dir(self)):
            return super().__setattr__(key, value)
        for att in self._attributes:
            if key.upper() == att.name:
                att.value = value
                return None
        raise AttributeError(f'{self.__class__.__name__} has no attribute {key}.')

    @override
    def __delattr__(self, key: str):
        if key.upper() in self.attributes:
            self._attributes = [att for att in self._attributes if att.name != key.upper()]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")

    def __contains__(self, x: str):
        return x.upper() in self.attributes

    @apply_to_each_input
    def apply(self, func, attr, *args, inplace=False, **kwargs):
        """Apply function to attributes.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept data as its first argument.
        attr : str, array-like
            Attributes to get data from.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        output : BaseComponent
            Transformed component.
        """
        data = getattr(self, attr)
        res = func(data, *args, **kwargs)
        if inplace:
            setattr(self, attr, res)
            return self
        return res

    def load(self, data, binary_data, logger):
        """Load data."""
        self._attributes = deepcopy(self._attributes_to_load)
        for attr in self._attributes:
            attr.component = self
            attr.load(data, binary_data, logger)


T = TypeVar('T', bound=BaseComponent)

class Attribute(Generic[T]):
    """Attribute."""
    def __init__(self,
                 name: str | None=None,
                 section: str | None=None,
                 kw: str | None=None,
                 custom_loader: AttributeLoaderType | None = None,
                 custom_ascii_loader=None,
                 postprocess: Callable[[Attribute[T]], None] | None=None,
                 not_present=None,
                 binary_file: resdp.binary.FileType | None=None,
                 binary_section=None,
                 binary_process=None,
                 sequential: bool=False):

        if name is not None:
            self.name: str = name
        else:
            if kw is None:
                raise ValueError('Either name or section should be provided.')
            self.name = kw

        self._custom_loader: AttributeLoaderType | None = custom_loader

        if custom_loader is not None:
            self._custom_ascii_loader = None
            if binary_file is not None:
                warnings.warn('`binary_file` argument is ignored when `custom_loader` is provided.')
            self._binary_file = None
            if binary_section is not None:
                warnings.warn('`binary_section` argument is ignored when `custom_loader` is provided.')
            self._binary_section = None
            if binary_process is not None:
                warnings.warn('`binary_process` argument is ignored when `custom_loader` is provided.')
            self._binary_process = None
            if postprocess is not None:
                warnings.warn('`postprocess` argument is ignored when `custom_loader` is provided.')
            self._postprocess = None
            if not_present is not None:
                warnings.warn('`not_present` argument is ignored when `custom_loader` is provided.')
            self._not_present = None
        else:
            self._custom_ascii_loader = custom_ascii_loader
            if (binary_file is None) != (binary_section is None):
                raise ValueError('Either both `binary_file` and `binary_section` are provided either none.')
            self._binary_file: resdp.binary.FileType | None = binary_file
            self._binary_section: str | None = binary_section
            self._binary_process = binary_process

            self._postprocess = postprocess
            self._not_present = not_present

        if custom_ascii_loader is not None or custom_loader is not None:
            self._kw = None
            self._section = None
        else:
            self._kw = kw
            self._section = section

        self._value = None
        self._component: Callable[[], T | None] | None = None
        self._sequential = sequential

    def _load_value(self, data, binary_data: resdp.binary.BinaryData, logger):
        if self.component is None:
            raise ValueError('Attribute should be associated with `BaseComponent` object.')
        if self._custom_loader is not None:
            val = self._custom_loader(data, binary_data, logger)
            self._value = val
            return self
        if self._binary_file is not None:
            val = self._load_ecl_binary_value(binary_data, logger)
        else:
            val = None
        if val is not None:
            self._value = val
            self.component.binary_attributes.append(self.name)
            return self
        if self._custom_ascii_loader is not None:
            self._value = self._custom_ascii_loader(data)
            return self
        if self._section in data:
            for entry in data[self._section]:
                if entry[0] == self._kw:
                    self._value = entry[1]
                    if self._sequential:
                        print(self.name, self._value)
                        self._value = np.array(self._value)[np.newaxis, :]
                    return self
        self._value = self._not_present
        return self

    def _load_ecl_binary_value(self, binary_data: resdp.binary.BinaryData | None, logger):
        _ = logger
        if binary_data is None:
            return None
        if self._binary_file is None:
            return None
        if self._binary_file not in binary_data:
            return None

        file_data = binary_data[self._binary_file]
        if self._binary_section is None:
            raise ValueError('`binary_file is specified but not `binary_section`.')
        if self._sequential:
            val = []
            while True:
                i = file_data.find(self._binary_section)
                if i is None:
                    break
                file_data.seek(i+1)
                val.append(file_data[i].value)
            if len(val) == 0:
                return None
            val = np.stack(val)
        else:
            i = file_data.find_unique(self._binary_section)
            if i is None:
                return None
            val = file_data[i].value
        if self._binary_process is not None:
            return self._binary_process(val)
        return val

    def load(self, data, binary_data, logger):
        """Load data."""
        self._load_value(data, binary_data, logger)
        if self._postprocess is not None:
            assert self._component is not None
            self._postprocess(self,)

    @property
    def value(self):
        """The value property."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def component(self) -> T | None:
        """Reference component."""
        if self._component is None:
            return None
        return self._component()

    @component.setter
    def component(self, value: T | None):
        if value is None:
            self._component = value
        else:
            self._component = ref(value)
