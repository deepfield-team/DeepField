"""BaseCompoment."""
from abc import abstractmethod
import os
from copy import deepcopy
from weakref import ref
import numpy as np
import h5py

from deepfield.field.parse_utils.ecl_binary import read_ecl_bin

from .utils import get_single_path

from .decorators import apply_to_each_input
from .parse_utils import read_array

class Attribute():
    def __init__(self, name=None, section=None, kw=None, custom_loader=None, postprocess=None, not_present=None, 
                 binary_file=None, binary_section=None, binary_process=None):

        if name is not None:
            self.name = name
        else:
            if kw is None:
                raise ValueError('Either name or section should be provided.')
            self.name = kw

        if custom_loader is not None:
            self._kw = None
            self._section = None
        else:
            if (section is None) or (kw is None):
                raise ValueError('Either both `section` and `kw` or `custom_loader` should be provided.')
            self._kw = kw
            self._section = section

        self._custom_loader = custom_loader
        self._postprocess = postprocess
        self._not_present = not_present
        self._value = None
        if (binary_file is None) != (binary_section is None):
            raise ValueError('Either both `binary_file` and `binary_section` are provided either none.')
        self._binary_file = binary_file
        self._binary_section = binary_section
        self._binary_process = binary_process

    def _load_value(self, data, path_to_results, basename, logger):
        __import__('ipdb').set_trace()
        if self._binary_file is not None:
            val = self._load_ecl_binary_value(path_to_results, basename, logger)
        else:
            val = None
        if val is not None:
            self._value = val
            return self
        if self._custom_loader is not None:
            self._value = self._custom_loader(data)
            return self
        for entry in data[self._section]:
            if entry[0] == self._kw:
                self._value = entry[1]
                return self
        self._value = self._not_present
        return self

    def _load_ecl_binary_value(self, path_to_results, basename, logger):
        path = get_single_path(path_to_results, basename + self._binary_file, logger)
        if path is None:
            return None
        attrs = [self._binary_section]
        sections = read_ecl_bin(path, attrs, logger=logger)
        if self._binary_section in sections:
            val = sections[self._binary_section]
            if self._binary_process is not None:
                return self._binary_process(val)
        else:
            return None

    def load(self, data, path_to_results, basename, logger):
        self._load_value(data, path_to_results, basename, logger)

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

    _attributes_to_load = []
    def __init__(self, *args, **kwargs):
        _ = args
        self._state = State()
        self._class_name = kwargs.pop('class_name', self.__class__.__name__)
        self._data = {}
        if 'field' in kwargs:
            self.field = kwargs['field']
        else:
            self.field = None
        for k, v in kwargs.items():
            if k != 'field':
                setattr(self, k, v)

    @property
    def field(self):
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
    def attributes(self):
        """Array of attributes."""
        return tuple(self._data.keys())

    @property
    def empty(self):
        """True if component is empty else False."""
        return not self._data

    def keys(self):
        """Array of attributes."""
        return self._data.keys()

    def values(self):
        """Returns a generator of attribute's data."""
        return self._data.values()

    def items(self):
        """Returns pairs of attribute's names and data."""
        return self._data.items()

    @property
    def state(self):
        """Get state."""
        return self._state

    @property
    def class_name(self):
        """Name of the component."""
        return self._class_name

    @class_name.setter
    def class_name(self, v):
        self._class_name = v

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
        if key.upper() in self._data:
            return self._data[key.upper()]
        raise AttributeError("{} has no attribute {}".format(self.class_name, key))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, key, value):
        if (key[0] == '_') or (key in dir(self)):
            return super().__setattr__(key, value)
        self._data[key.upper()] = value
        return self

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __delattr__(self, key):
        if key.upper() in self._data:
            del self._data[key.upper()]
        else:
            raise AttributeError("{} has no attribute {}".format(self.class_name, key))

    def __delitem__(self, key):
        return delattr(self, key)

    def __contains__(self, x):
        return x.upper() in self.attributes

    def copy(self):
        """Returns a deepcopy of attributes. Cached properties are not copied."""
        copy = self.__class__(
            **{k: deepcopy(v) if not issubclass(v.__class__, BaseComponent) else v.copy() for k, v in self.items()},
            field = self.field,
        )
        copy.set_state(**self.state.as_dict())
        copy.class_name = self.class_name
        return copy

    def drop(self, attr):
        """Drop an attribute."""
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

    def load(self, path_or_buffer, **kwargs):
        """Load data from a file or buffer.

        Parameters
        ----------
        path_or_buffer : str of string buffer
            Source to read data from.
        **kwargs : dict, optional
            Any kwargs to be passed to load method.

        Returns
        -------
        comp : BaseComponent
            BaseComponent with loaded attributes.
        """
        __import__('ipdb').set_trace()
        if isinstance(path_or_buffer, str):
            if os.path.isdir(path_or_buffer):
                return self._load_ecl_binary(path_or_buffer, **kwargs)
            name = os.path.basename(path_or_buffer)
            fmt = os.path.splitext(name)[1].strip('.')
            return self._get_fmt_loader(fmt)(path_or_buffer, **kwargs)
        return self._read_buffer(path_or_buffer, **kwargs)

    def load(self, data, path_to_results, basename, logger):
        """Load data."""
        self._attributes = deepcopy(self._attributes_to_load)
        for attr in self._attributes:
            attr.load(data, path_to_results, basename, logger)


    def _load_ecl_binary(self, path_to_results, **kwargs):
        """Load data from RESULTS derictory."""
        raise NotImplementedError('Load from binary files is not implemented.')

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

    def _read_buffer(self, buffer, attr, logger=None, **kwargs):
        """Read array-like data from string buffer.

        Parameters
        ----------
        buffer : buffer
            String buffer to read from.
        attr : str
            Target attribute.
        logger : logger
            Event logger.
        kwargs : misc
            Any additional named arguments to ``read_array``.

        Returns
        -------
        comp : BaseComponent
            BaseComponent with new attribute.
        """
        _ = logger
        arr = read_array(buffer, **kwargs)
        setattr(self, attr, arr)
        return self

    def dump(self, path, **kwargs):
        """Dump attributes into file.

        Parameters
        ----------
        path : str
            Path to output file.
        kwargs : dict, optional
            Any kwargs for dump method.

        Returns
        -------
        comp : BaseComponent
            BaseComponent unchanged.
        """
        fname = os.path.basename(path)
        fmt = os.path.splitext(fname)[1].strip('.')

        if fmt.upper() == 'HDF5':
            self._dump_hdf5(path, **kwargs)
        elif fmt.upper() in ['DAT', 'DATA', 'INC', 'GRDECL']:
            self._dump_ascii(path, **kwargs)
        else:
            raise NotImplementedError('File format {} not supported.'.format(fmt))
        return self

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

    def _dump_ascii(self, path, attrs=None, mode='w', fmt='%f', compressed=True, **kwargs):  # pylint: disable=too-many-branches
        """Save array-like data into ASCII file.

        Parameters
        ----------
        path : str
            Path to output file.
        attrs : str, array of str or None
            Array of keywords to dump into file.
        mode : str
            Mode to open file.
            'w': write, a new file is created (an existing file with
            the same name would be deleted).
            'a': append, an existing file is opened for reading and writing,
            and if the file does not exist it is created.
            Default to 'w'.
        fmt : str or sequence of strs, optional
            Format to be passed into ``numpy.savetxt`` function. Default to '%f'.
        kwargs : misc
            Kwargs for `_make_data_dump`.

        Returns
        -------
        comp : BaseComponent
            BaseComponent unchanged.
        """
        if attrs is None:
            attrs = self.attributes
        elif isinstance(attrs, str):
            attrs = [attrs]
        with open(path, mode) as f:
            for attr in attrs:
                data = self._make_data_dump(attr, fmt='ascii', **kwargs)
                if data.dtype == bool:
                    data = data.astype(int)
                self.dump_array_ascii(f, data, header=attr.upper(),
                                      fmt=fmt, compressed=compressed)
        return self

        @staticmethod
        def dump_array_ascii(buffer, array, header=None, fmt='%f', compressed=True):
            """Writes array-like data into an ASCII buffer.

            Parameters
            ----------
            buffer : buffer-like
            array : 1d, array-like
                Array to be saved
            header : str, optional
                String to be written line before the array
            fmt : str or sequence of strs, optional
                Format to be passed into ``numpy.savetxt`` function. Default to '%f'.
            compressed : bool
                If True, uses compressed typing style
            """
            if header is not None:
                buffer.write(header + '\n')

            if compressed:
                i = 0
                items_written = 0
                while i < len(array):
                    count = 1
                    while (i + count < len(array)) and (array[i + count] == array[i]):
                        count += 1
                    if count <= 4:
                        buffer.write(' '.join([fmt % array[i]] * count))
                        items_written += count
                    else:
                        buffer.write(str(count) + '*' + fmt % array[i])
                        items_written += 1
                    i += count
                    if items_written > MAX_STRLEN:
                        buffer.write('\n')
                        items_written = 0
                    else:
                        buffer.write(' ')
                buffer.write('/\n')
            else:
                for i in range(0, len(array), MAX_STRLEN):
                    buffer.write(' '.join([fmt % d for d in array[i:i + MAX_STRLEN]]))
                    buffer.write('\n')
                buffer.write('/\n')
