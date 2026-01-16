"""BaseCompoment."""
from __future__ import annotations
import os
from copy import deepcopy
import pdb
from weakref import ref
import numpy as np
import h5py

from .decorators import apply_to_each_input
from .parse_utils import read_array

from typing import TYPE_CHECKING, Callable, Sequence, TypedDict, override

if TYPE_CHECKING:
    from .field import Field
    import resdp.binary

class DumpDict(TypedDict):
    attributes: Sequence[Attribute]
    state: State
    field: Field | None

class Attribute():
    def __init__(self,
                 name: str | None=None,
                 section: str | None=None,
                 kw: str | None=None,
                 custom_loader=None,
                 postprocess=None,
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

        if custom_loader is not None:
            self._kw = None
            self._section = None
        else:
            self._kw = kw
            self._section = section

        self._custom_loader = custom_loader
        self._postprocess = postprocess
        self._not_present = not_present
        self._value = None
        if (binary_file is None) != (binary_section is None):
            raise ValueError('Either both `binary_file` and `binary_section` are provided either none.')
        self._binary_file: resdp.binary.FileType | None = binary_file
        self._binary_section: str | None = binary_section
        self._binary_process = binary_process
        self._component: Callable[[], BaseComponent | None] | None = None
        self._sequential = sequential

    def _load_value(self, data, binary_data: resdp.binary.BinaryData, logger):
        if self.component is None:
            raise ValueError('Attribute should be associated with `BaseComponent` object.')
        if self._binary_file is not None:
            val = self._load_ecl_binary_value(binary_data, logger)
        else:
            val = None
        if val is not None:
            self._value = val
            self.component.state.binary_attributes.append(self.name)
            return self
        if self._custom_loader is not None:
            self._value = self._custom_loader(data)
            return self
        if self._section in data:
            for entry in data[self._section]:
                if entry[0] == self._kw:
                    self._value = entry[1]
                    return self
        self._value = self._not_present
        return self

    def _load_ecl_binary_value(self, binary_data: resdp.binary.BinaryData | None, logger):
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
        self._load_value(data, binary_data, logger)

    @property
    def value(self):
        """The value property."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
    @property
    def component(self) -> BaseComponent | None:
        if self._component is None:
            return None
        else:
            return self._component()
    @component.setter
    def component(self, value: BaseComponent | None):
        if value is None:
            self._component = value
            return None
        self._component = ref(value)

MAX_STRLEN = 40

class State:
    """State holder."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if 'binary_attributes' not in kwargs:
            self.binary_attributes = []

    def as_dict(self):
        """Dict of states."""
        return self.__dict__

    def __repr__(self):
        return repr(self.__dict__)


class BaseComponent:
    """Base class for components of geological model."""

    _attributes_to_load: list[Attribute] = []
    def __init__(self, dump=None, field=None):
        self._field = None
        if dump is not None:
            self._attributes = dump['attributes']
            self.field = dump['field']
            self._state = dump['state']
            for att in self._attributes:
                att.component = self
            return None
        self._attributes: list[Attribute] = []
        self._state = State()
        self.field = field

    @property
    def field(self) -> Field:
        """Field associated with the component."""
        return self._field()

    @field.setter
    def field(self, field):
        """Set field to which component belongs."""
        if isinstance(field, ref) or field is None:
            self._field = field
            return self
        self._field = ref(field)
        return self

    @property
    def attributes(self) -> Sequence[str]:
        """Array of attributes."""
        return tuple((attr.name for attr in self._attributes if attr.value is not None))

    @property
    def empty(self):
        """True if component is empty else False."""
        return not self._attributes

    def keys(self):
        """Array of attributes."""
        return (attr.name for attr in self._attributes)

    def values(self):
        """Returns a generator of attribute's data."""
        return (attr.value for attr in self._attributes)

    def items(self):
        """Returns pairs of attribute's names and data."""
        return ((attr.name, attr.value) for attr in self._attributes)

    @property
    def state(self):
        """Get state."""
        return self._state

    @property
    def class_name(self):
        """Name of the component."""
        return self.__class__.__name__

    def empty_like(self):
        """Get an empty component with the same state and the structure of embedded BaseComponents (if any)."""
        empty = BaseComponent(class_name=self.class_name)
        for comp, value in self.items():
            if issubclass(value.__class__, BaseComponent):
                empty[comp] = value.empty_like()
        empty.set_state(**self.state.as_dict())
        return empty

    def set_state(self, **kwargs):
        """State setter."""
        for k, v in kwargs.items():
            setattr(self.state, k, v)
        return self

    def del_state(self, *args):
        """State remover."""
        for k in args:
            if not hasattr(self.state, k):
                raise AttributeError('{} has no state {}'.format(self.class_name, k))
            delattr(self.state, k)
        return self

    def __getattr__(self, key):
        for attr in self._attributes:
            if key.upper() == attr.name:
                return attr.value
        raise AttributeError("{} has no attribute {}".format(self.class_name, key))
    def dump_dict(self) -> DumpDict:
        return {
            'attributes': deepcopy(self._attributes),
            'field': self.field,
            'state': deepcopy(self.state)
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, key, value):
        if (key[0] == '_') or (key in dir(self)):
            return super().__setattr__(key, value)
        for att in self._attributes:
            if key.upper() == att.name:
                att.value = value
                return None
        raise AttributeError(f'{self.class_name} has no attribute {key}.')

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @override
    def __delattr__(self, key: str):
        if key.upper() in self.attributes:
            self._attributes = [att for att in self._attributes if att.name != key.upper()]
        else:
            raise AttributeError(f"{self.class_name} has no attribute {key}")

    def __delitem__(self, key: str):
        return delattr(self, key)

    def __contains__(self, x: str):
        return x.upper() in self.attributes

    def copy(self):
        """Returns a deepcopy of attributes. Cached properties are not copied."""
        copy = self.__class__(
           dump=self.dump_dict()
        )
        return copy

    def drop(self, attr):
        """Drop an attribute."""
        raise NotImplementedError()
        del self._data[attr.upper()]
        return self

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

    @apply_to_each_input
    def reshape(self, attr, newshape, order='C', inplace=True):
        """Reshape `numpy.ndarray` attributes.

        Parameters
        ----------
        attr : str, array of str
            Attribute to be reshaped.
        newshape : tuple
            New shape.
        order : str
            Numpy reshape order. Default to 'C'.
        inplace : bool
            If `True`, reshape is made inplace, return BaseComponent.
            Else, return reshaped attribute.

        Returns
        -------
        output : BaseComponent if inplace else reshaped attribute itself.
        """
        data = getattr(self, attr)
        if data is None:
            return None
        if isinstance(data, np.ndarray) and data.ndim:
            data = np.reshape(data, newshape, order=order)
        elif hasattr(data, 'reshape'):
            data = data.reshape(newshape, order=order)
        else:
            raise ValueError('Attribute {} can not be reshaped.'.format(attr))
        if inplace:
            setattr(self, attr, data)
            return self
        return data

    def ravel(self, attr=None, order='F'):
        """Ravel attributes where applicable assuming by default Fortran order.

        Parameters
        ----------
        attr : str, array of str
            Attribute to ravel.
        order : str
            Numpy reshape order. Default to 'F'.

        Returns
        -------
        out : Raveled attribute.
        """
        return self.reshape(attr=attr, newshape=(-1, ), order=order, inplace=False)

    def _get_fmt_loader(self, fmt):
        """Get loader for given file format."""
        if fmt.upper() == 'HDF5':
            return self._load_hdf5
        raise NotImplementedError('File format .%s is not supported.' % fmt.upper())

    def add_attribute(self, att: Attribute):
        att.component = self
        self._attributes.append(att)

    def load(self, data, binary_data, logger):
        """Load data."""
        self._attributes = deepcopy(self._attributes_to_load)
        for attr in self._attributes:
            attr.component = self
            attr.load(data, binary_data, logger)

    def _load_hdf5(self, path, attrs=None, raise_errors=False, logger=None, subset=None, **kwargs):
        """Load data from a HDF5 file.

        Parameters
        ----------
        path : str
            Path to file to load data from.
        attrs : str or array of str, optional
            Array of dataset's names to get from file. If not given, loads all.
        raise_errors : bool
            Errors behaviour. If True missing attributes in HDF5 file will raise an error.
            If False, missing attributes in HDF5 file will be ignored.
        logger : logger
            Event logger.
        subset : slice or list of indices
            Subset of items to load. Be default all items are loaded.

        Returns
        -------
        comp : BaseComponent
            BaseComponent with loaded attributes.
        """
        raise NotImplementedError()
        _ = kwargs
        if isinstance(attrs, str):
            attrs = [attrs]
        if subset is None:
            subset = ()
        with h5py.File(path, 'r') as f:
            self._load_hdf5_group(f, attrs=attrs, raise_errors=raise_errors, logger=logger, subset=subset)
        return self

    def _load_hdf5_group(self, grp, attrs, raise_errors, logger, subset):
        """Load data from a group from an hdf5 file. Recursively runs itself when finds a nested group.

        Parameters
        ----------
        grp : h5py.Group
            A group to load self from.
        attrs : array-like of str
            Array of dataset's names to get from file. If not given, loads all.
        raise_errors : bool
            Errors behaviour. If True missing attributes in HDF5 file will raise an error.
            If False, missing attributes in HDF5 file will be ignored.
        logger : logger
            Event logger.
        subset : slice or list of indices
            Subset of items to load. Be default all items are loaded.
        """
        raise NotImplementedError()
        grp = grp[self.class_name]
        state = {k : v for k, v in grp.attrs.items() if k!='DATES'}
        for k, v in state.items():
            try:
                state[k] = v if not np.isnan(v) else None
            except TypeError:
                state[k] = v
        self.set_state(**state)
        for att in grp.keys() if attrs is None else attrs:
            try:
                val = grp[att.upper()]
            except KeyError as err:
                if raise_errors:
                    raise err
                if logger is not None:
                    logger.info('Attribute %s not found in %s.' % (att.upper(), grp.name))
                continue
            if isinstance(val, h5py.Group):
                val = BaseComponent(class_name=att)
                val._load_hdf5_group(grp, attrs, raise_errors, logger, subset)  # pylint: disable=protected-access
            else:
                val = val[subset]
                if val.size == 1:
                    val = val[0]
            setattr(self, att, val)

    def _make_data_dump(self, attr, fmt=None, **kwargs):
        """Prepare data for dump."""
        _ = fmt, kwargs
        return getattr(self, attr)

    def _dump_hdf5(self, path, mode='a', compression=None, state=False, **kwargs):
        """Save data into HDF5 file.

        Parameters
        ----------
        path : str
            Path to output file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'a'.
        compression : str
            Compression method. If None, no compression is applied.
        state : bool
            Dump compoments's state.
        kwargs : misc
            Kwargs for `_make_data_dump`.

        Returns
        -------
        comp : BaseComponent
            BaseComponent unchanged.
        """
        with h5py.File(path, mode) as f:
            self._dump_hdf5_group(f, compression=compression, state=state, **kwargs)
        return self

    def _dump_hdf5_group(self, grp, compression, state, **kwargs):
        """Save BaseComponent into a group of HDF5 file. If BaseComponent have nested BaseComponents as attributes,
        saves them to nested groups recursively.

        Parameters
        ----------
        grp : h5py.Group
            Path to output file.
        compression : str
            Compression method. If None, no compression is applied.
        state : bool
            Dump compoments's state.
        kwargs : misc
            Kwargs for `_make_data_dump`.
        """
        grp = grp[self.class_name] if self.class_name in grp else grp.create_group(self.class_name)
        if state:
            for k, v in self.state.as_dict().items():
                grp.attrs[k] = v if v is not None else np.nan
        for att, value in self.items():
            if issubclass(value.__class__, BaseComponent):
                value._dump_hdf5_group(grp, compression=compression, state=state, **kwargs)  # pylint: disable=protected-access
            else:
                data = self._make_data_dump(att, fmt='hdf5', **kwargs)
                if att in grp:
                    del grp[att]
                grp.create_dataset(att, data=data, compression=compression)

