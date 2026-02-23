import logging
from typing import cast
from resdp import DataType
from resdp.binary import BinaryData
import pandas as pd


def load_welltrack(data: DataType, binary_data: BinaryData, logger: logging.Logger) -> pd.DataFrame | None:
    _ = binary_data, logger
    section = 'SCHEDULE'
    res: list[pd.DataFrame] = []
    if not section in data:
        return None
    for key, val in data[section]:
        if key == 'WELLTRACK':
            assert isinstance(val, tuple)
            assert len(val) == 2
            assert isinstance(val[0], str)
            assert isinstance(val[1], pd.DataFrame)
            res.append(cast(pd.DataFrame, val[1]).assign(WELL=cast(str, val[0])))
    if not res:
        return None
    return pd.concat(res)
