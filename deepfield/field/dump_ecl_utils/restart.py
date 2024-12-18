"""RESTART dump assisting functions."""
import os
import numpy as np

from .share import (ARRAYMAX, format_keyword, ARRAYMIN, DOUBHEAD, ENDSOL, ICON, IGRP, INTEHEAD,
                    ITIME, IWEL, LOGIHEAD, META_BLOCK_SPEC, NAME, NUMBER, POINTER, POINTERB, REAL,
                    SPEC_DATA_BLOC_SPEC, STARTSOL, SpecElement, TIME, TYPE, ZWEL,
                    write_unrst_data_section, write_unrst_data_section_f,
                    write_unrst_section, split_pointer)

from ..parse_utils.ecl_binary import _read_sections

# Header's block formats
INDEX_SECTIONS_DATA_START = {
    INTEHEAD: SpecElement(el_struct=META_BLOCK_SPEC[INTEHEAD]),
    LOGIHEAD: SpecElement(el_struct=META_BLOCK_SPEC[LOGIHEAD]),
    DOUBHEAD: SpecElement(el_struct=META_BLOCK_SPEC[DOUBHEAD]),
    IGRP: SpecElement(el_struct=META_BLOCK_SPEC[IGRP]),
    IWEL: SpecElement(el_struct=META_BLOCK_SPEC[IWEL]),
    ZWEL: SpecElement(el_struct=META_BLOCK_SPEC[ZWEL]),
    ICON: SpecElement(el_struct=META_BLOCK_SPEC[ICON]),
    STARTSOL: SpecElement(el_struct=META_BLOCK_SPEC[STARTSOL])
}
INDEX_SECTIONS_DATA_END = {
    ENDSOL: SpecElement(el_struct=META_BLOCK_SPEC[ENDSOL])
}

RESTART_EXT = '.X{time:0>4}'
UNIFIED_RESTART_EXT = '.UNRST'
INDEX_RESTART_EXT = '.RSSPEC'


def get_restart_file(is_unified, name, time, unified_file, mode):
    """
    Get file object for restart file.

    Parameters
    ----------
    is_unified : bool
        If true, the target file will be in unified format.
        Otherwise the target file will be in non unified format.
    time: int
        Time

    Returns
    -------
    out : file object

    Notes
    -----
    If is_unified is True the function return the same file.
    If is_unified is False the function return the new file each time.

    """
    if is_unified:
        if not unified_file:
            unified_file = open(name + UNIFIED_RESTART_EXT, f'{mode}+b') #pylint: disable=consider-using-with
        return unified_file, unified_file

    return open(name + RESTART_EXT.format(time=time), f'{mode}+b'), unified_file#pylint: disable=consider-using-with


class PointerStruct:
    """Pointer struct."""
    def __init__(self, pointer_offset, size_inte, size_logi, size_doub, size_igrp,
                 size_iwel, size_zwel, size_icon, size_smes, size_emes,
                 **size_states):
        ptr_dict = {}
        ptr_dict[INTEHEAD] = pointer_offset
        ptr_dict[LOGIHEAD] = ptr_dict[INTEHEAD] + size_inte
        ptr_dict[DOUBHEAD] = ptr_dict[LOGIHEAD] + size_logi
        ptr_dict[IGRP] = ptr_dict[DOUBHEAD] + size_doub
        ptr_dict[IWEL] = ptr_dict[IGRP] + size_igrp
        ptr_dict[ZWEL] = ptr_dict[IWEL] + size_iwel
        ptr_dict[ICON] = ptr_dict[ZWEL] + size_zwel
        ptr_dict[STARTSOL] = ptr_dict[ICON] + size_icon
        pos = ptr_dict[STARTSOL] + size_smes
        for state, size in size_states.items():
            ptr_dict[state.upper()] = pos
            pos = ptr_dict[state.upper()] + size
        ptr_dict[ENDSOL] = pos

        self.ptrs = ptr_dict
        self.size = (size_inte + size_logi + size_doub + size_igrp + size_iwel + size_zwel +
                     size_icon + size_smes + sum(size_states.values()) + size_emes)


class Pointers:
    """Pointers."""
    def __init__(self, pointers, key_list):
        splited_pointers = list(map(lambda key: split_pointer(pointers.ptrs[key]), key_list))
        self.pointer_a = np.array(list(map(lambda splited_pointer: splited_pointer[0], splited_pointers)))
        self.pointer_b = np.array(list(map(lambda splited_pointer: splited_pointer[1], splited_pointers)))

def save_restart(is_unified, name, data, attrs, dates, grid_dim, time_size, mode, logger=None):
    """
    Function for saving target RESTART and RSSPEC data files

    Parameters
    ----------
    is_unified : bool
        If true, the target file will be in unified format.
        Otherwise the target file will be in non unified format.
    """
    def logger_print(msg, level='warning'):
        if logger is not None:
            getattr(logger, level)(msg)

    index_sections_data = {}
    index_sections_data.update(INDEX_SECTIONS_DATA_START)
    index_sections_data.update({k.upper(): SpecElement(k.upper(), REAL) for k in attrs})
    index_sections_data.update(INDEX_SECTIONS_DATA_END)

    data_block_spec = {k.upper(): SpecElement(k.upper(), REAL) for k in attrs}

    start_idx = 1
    start_t = 0
    if (mode == 'w' or\
        (mode == 'a' and\
         (not os.path.isfile(name + '.UNRST') or not os.path.isfile(name + '.RSSPEC')))):
        mode = 'w'
        start_idx = 0
    elif os.path.exists(name + '.RSSPEC'):
        start_t = len(_read_sections(name + '.RSSPEC', decode=False, sequential=True)[1]['TIME']) - 1
    else:
        logger_print("Dumping restart: No RSSPEC file found")

    unified_file = None
    pointer_offset = 0

    with open(name + INDEX_RESTART_EXT, f'{mode}+b') as file_index:
        if mode == 'w':
            write_unrst_section(file_index, INTEHEAD, META_BLOCK_SPEC[INTEHEAD], grid_dim)

        for time in range(start_idx, time_size):
            file_restart, unified_file = get_restart_file(is_unified, name, start_t + time, unified_file, mode)
            pointers = save_restart_bytime(file_restart, index_sections_data, data_block_spec,
                                           grid_dim, data, attrs, time, pointer_offset)

            save_index_section_bytime(file_index, index_sections_data, data_block_spec,
                                      pointers, dates[time].date(), start_t + time)

            if is_unified:
                pointer_offset += pointers.size
            else:
                file_restart.close()

    if unified_file and not unified_file.closed:
        unified_file.close()

    return pointer_offset

def get_itime_section_data(date, time):
    """
    Get data for ITIME section

    Parameters
    ----------
    time : int
        Time interval

    Returns
    -------
    out : np.array
        Array of items

    Examples
    --------
    >>> l = get_itime_section_data(5)
    >>> l
    array([    5,     6,     1,  2019, -2345,     1,     0, -2345, -2345, -2345,     0,     0,     0])

    """
    return np.array([time, date.day, date.month, date.year, -2345, 1, 0, -2345, -2345, -2345, 0, 0, 0])

def get_type_section_data(index_sections_data, key_list):
    """
    Get type list for TYPE section in INSPEC target file

    Parameters
    ----------
    key_list : list
        The list of keys for which you want to create a list of types

    Returns
    -------
    out : np.array
        Type list for given keys

    Examples
    --------
    >>> l = get_type_section_data(['INTEHEAD', 'LOGIHEAD'])
    >>> l
    array(['INTE    ', 'LOGI    '], dtype='<U8')

    """
    return np.array(list(map(lambda key: format_keyword(index_sections_data[key].type), key_list)))

def get_number_section_data(index_sections_data, key_list):
    """
    Get number list of items for target sections

    Parameters
    ----------
    key_list : list
        The list of keys for which you want to get number of items

    Returns
    -------
    out : np.array
        Number list of items in target sections

    Examples
    --------
    >>> l = get_number_section_data(['INTEHEAD', 'LOGIHEAD'])
    >>> l
    array([296, 100])

    """
    return np.array(list(map(lambda key: index_sections_data[key].number, key_list)))

def get_pointer_section_data(key_list, pointers):
    """
    Get offsets if target sections in INIT file

    Parameters
    ----------
    key_list : list
        List of keys for which you want to get number of items
    pointers : PointerStruct
        List of offsets of the specified sections in target INIT file

    Returns
    -------
    out : np.array
        List of offsets of offsets of target sections

    Examples
    --------
    >>> l = get_pointer_section_data(['INTEHEAD', 'LOGIHEAD'], pointers)
    >>> l
    array([43, 432])

    """
    return Pointers(pointers, key_list)

def get_arraymax_section_data(data_block_spec, key_list, max_size=1300000000):
    """
    List of array max parameter of target sections in INIT file

    Parameters
    ----------
    key_list : list
        List of keys for which you want to get number of items
    max_size : int, optional
        Max size, default 1300000000

    Returns
    -------
    out : np.array
        List of array max parameter of target sections

    Examples
    --------
    >>> l = get_arraymax_section_data(['INTEHEAD', 'LOGIHEAD'])
    >>> l
    array([1300000000, 1300000000])

    """
    result_list = []
    data_key_list = data_block_spec.keys()
    for key in key_list:
        if key in data_key_list:
            result_list.append(max_size)
        else:
            result_list.append(0)
    return np.array(result_list)

def save_index_section_bytime(f, index_sections_data, data_block_spec, pointers, date, time):
    """
    Save index section by time segment

    Parameters
    ----------
    f : file object
    pointers : PointerStruct
    time : int
        Time interval

    See Also
    --------
    PointerStruct

    """
    key_list = list(index_sections_data.keys())
    write_unrst_data_section(f=f, name=TIME, stype=SPEC_DATA_BLOC_SPEC[TIME],
                             data_array=np.array([time]))
    write_unrst_data_section(f=f, name=ITIME, stype=SPEC_DATA_BLOC_SPEC[ITIME],
                             data_array=get_itime_section_data(date, time))
    write_unrst_data_section(f=f, name=NAME, stype=SPEC_DATA_BLOC_SPEC[NAME],
                             data_array=np.array(list(map(format_keyword, key_list))))
    write_unrst_data_section(f=f, name=TYPE, stype=SPEC_DATA_BLOC_SPEC[TYPE],
                             data_array=get_type_section_data(index_sections_data, key_list))
    write_unrst_data_section(f=f, name=NUMBER, stype=SPEC_DATA_BLOC_SPEC[NUMBER],
                             data_array=get_number_section_data(index_sections_data, key_list))
    pointers = get_pointer_section_data(key_list, pointers)
    write_unrst_data_section(f=f, name=POINTER, stype=SPEC_DATA_BLOC_SPEC[POINTER],
                             data_array=pointers.pointer_a)
    write_unrst_data_section(f=f, name=POINTERB, stype=SPEC_DATA_BLOC_SPEC[POINTERB],
                             data_array=pointers.pointer_b)
    write_unrst_data_section(f=f, name=ARRAYMAX, stype=SPEC_DATA_BLOC_SPEC[ARRAYMAX],
                             data_array=get_arraymax_section_data(data_block_spec, key_list))
    write_unrst_data_section(f=f, name=ARRAYMIN, stype=SPEC_DATA_BLOC_SPEC[ARRAYMIN],
                             data_array=np.array([0] * len(key_list)))

def save_restart_bytime(f, index_sections_data, data_block_spec,
                        grid_dim, states, attrs, time, pointer_offset):
    """
    Save restart section by time segment

    Parameters
    ----------
    f : file object
    time : int
        Time interval

    """
    data = [s[time].ravel(order='F') for s in states]

    size_inte = write_unrst_section(f, INTEHEAD, META_BLOCK_SPEC[INTEHEAD], grid_dim)
    size_logi = write_unrst_section(f, LOGIHEAD, META_BLOCK_SPEC[LOGIHEAD])
    size_doub = write_unrst_section(f, DOUBHEAD, META_BLOCK_SPEC[DOUBHEAD])
    size_igrp = write_unrst_section(f, IGRP, META_BLOCK_SPEC[IGRP])
    size_iwel = write_unrst_section(f, IWEL, META_BLOCK_SPEC[IWEL])
    size_zwel = write_unrst_section(f, ZWEL, META_BLOCK_SPEC[ZWEL])
    size_icon = write_unrst_section(f, ICON, META_BLOCK_SPEC[ICON])
    size_smes = write_unrst_section(f, STARTSOL, META_BLOCK_SPEC[STARTSOL])

    sizes = {}
    for i, attr in enumerate(attrs):
        attr = attr.upper()
        items = int(np.prod(data[i].shape))
        index_sections_data[attr].number = items
        size = write_unrst_data_section_f(f=f,
                                          name=attr,
                                          stype=data_block_spec[attr].type,
                                          data_array=data[i])
        sizes[attr] = size

    size_emes = write_unrst_section(f, ENDSOL, META_BLOCK_SPEC[ENDSOL])
    return PointerStruct(pointer_offset, size_inte, size_logi, size_doub, size_igrp,
                         size_iwel, size_zwel, size_icon, size_smes, size_emes,
                         **sizes)
