"""Decorators."""
from functools import wraps
import numpy as np
from anytree import PreOrderIter


class cached_property:  # pylint: disable=invalid-name
    """Cached property decorator.

    May be used to apply additional transformations to the cached variable.

    Parameters
    ----------
    arg : function
        If the decorator is used without arguments, represents the decorated method.
        Else, represents additional transformation applied to the output of the decorated method.
        In the latter case, should have the following interface:
            arg : instance, output -> modified_output
        where `instance` is the instance of the method's class.
    modify_cache : bool, optional
        If True, applies transformation not only to the output but rather to the cached variable itself.
    """
    def __init__(self, arg, modify_cache=False):
        self._update_property(arg)
        self.modify_cache = modify_cache
        self.apply_to_output = lambda instance, out: out

    def __call__(self, arg):
        self.apply_to_output = self.property
        self._update_property(arg)
        return self

    def __get__(self, instance, cls=None):
        if self.name in instance.__dict__:
            return self._get_from_cache(instance)
        return self._compute_and_store(instance)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

    def _update_property(self, arg):
        """Update property, its name and docstring given new property candidate."""
        self.property = arg
        self.name = arg.__name__
        self.__doc__ = arg.__doc__

    def _get_from_cache(self, instance):
        """Loads data from the cache. Modifies cache if required."""
        data = instance.__dict__[self.name]
        data = self.apply_to_output(instance, data)
        if self.modify_cache:
            instance.__dict__[self.name] = data
        return data

    def _compute_and_store(self, instance):
        """Computes data using decorated method. Stores it in the cache (modified, if required)."""
        data = self.property(instance)
        if self.modify_cache:
            data = self.apply_to_output(instance, data)
            instance.__dict__[self.name] = data
        else:
            instance.__dict__[self.name] = data
            data = self.apply_to_output(instance, data)
        return data


def apply_to_each_input(method):
    """Apply the method to each input if array of inputs is given.
    If inputs are not specified, apply to each of self.attributes.
    """
    @wraps(method)
    def decorator(self, *args, attr=None, **kwargs):
        """Returned decorator."""
        is_list = True
        if isinstance(attr, str):
            attr = (attr, )
            is_list = False
        elif attr is None:
            attr = self.attributes
            if not self.attributes:
                return self

        res = []
        for att in attr:
            res.append(method(self, *args, attr=att.upper(), **kwargs))
        if isinstance(res[0], self.__class__):
            return self
        return res if is_list else res[0]
    return decorator

def apply_to_each_node(method, include_groups=False):
    """Apply a method to each tree node.

    Parameters
    ----------
    method : callable
        Method to be decorated. Node should be second argument of the method.
    include_groups : bool, optional
        If False, group nodes are not evaluated. Default to False.

    Returns
    -------
    decorator : callable
        Decorated method.
    """

    @wraps(method)
    def decorator(self, *args, **kwargs):
        """Returned decorator."""
        res = []
        for node in PreOrderIter(self.root):
            if (not node.is_group) or include_groups:
                res.append(method(self, node, *args, **kwargs))
        if not res:
            return self
        if isinstance(res[0], self.__class__):
            return self
        return np.array(res)

    return decorator
