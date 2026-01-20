import logging
from tkinter import W
from typing import cast
import warnings
from numpy.typing import NDArray
import numpy as np
import resdp
import resdp.binary
import pandas as pd


def load_results(data: resdp.DataType,
                 binary_data: resdp.binary.BinaryData,
                 logger: logging.Logger) -> pd.DataFrame | None:
    _ = data, logger
    if 'SMSPEC' not in binary_data:
        return None
    if 'UNSMRY' not in binary_data:
        return None
    smspec_data = binary_data['SMSPEC']
    unsmry_data = binary_data['UNSMRY']

    i = smspec_data.find('KEYWORDS')
    if i is None:
        warnings.warn('Could not find section `KEYWORDS` in `.SMSPEC` file.')
        return None
    keywords = np.char.strip(cast(NDArray[np.str_], smspec_data[i].value))
    indices_to_keep: list[int] = []

    keywords_to_keep: list[str] = []

    for i, kw in enumerate(keywords):
        kw = cast(str, kw)
        kw = kw.strip()
        if  kw.startswith('W') or kw in ('DAY', 'MONTH', 'YEAR'):
            indices_to_keep.append(i)
            keywords_to_keep.append(kw)
    i = None
    i = smspec_data.find('WGNAMES')
    if i is None:
        warnings.warn('Could not find section `WGNAMES` in `.SMSPEC` file.')
        return None
    wgnames = cast(NDArray[np.str_], smspec_data[i].value)
    wgnames = np.char.strip(wgnames[indices_to_keep])

    data: list[NDArray[np.float_]] = []
    while True:
        i = unsmry_data.find('PARAMS')
        if i is None:
            break
        data.append(cast(NDArray[np.float_], unsmry_data[i].value[indices_to_keep]))
        if i+1 < len(unsmry_data):
            unsmry_data.seek(i+1)
        else:
            break

    data = np.stack(data)
    name_placeholder: str = wgnames[keywords[indices_to_keep]=='YEAR'][0]
    well_names = np.unique(wgnames[wgnames!=name_placeholder])

    df = pd.DataFrame()
    dates = pd.to_datetime(
        {
            'year': np.repeat(data[:, keywords[indices_to_keep]=='YEAR'], well_names.size),
            'month': np.repeat(data[:, keywords[indices_to_keep]=='MONTH'], well_names.size),
            'day': np.repeat(data[:, keywords[indices_to_keep]=='DAY'], well_names.size)
        }
    )
    df['DATE'] = dates
    df['WELL'] = np.tile(well_names, data.shape[0])
    for kw in np.unique(keywords[indices_to_keep]):
        if kw not in ('MONTH', 'YEAR', 'DAY'):
            df[kw] = np.nan
        for wn in well_names:
            ind = ((wgnames == wn) & (keywords[indices_to_keep] == kw))
            if not ind.any():
                break
            if sum(ind) > 1:
                raise ValueError(f'Several values for keyword `{kw}` and well `{wn}`.')
            df.loc[df['WELL']==wn, kw] = data[:, ind]
    return df

