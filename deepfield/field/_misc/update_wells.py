from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ..wells import WellScheduleAttribute, Wells


def update_wells(att: WellScheduleAttribute[Wells]):
    if att.value is None:
        return
    val = att.value
    component = att.component
    assert component is not None
    assert isinstance(val, pd.DataFrame)
    if not val.empty:
        welldata = {}
        for k, v in val.groupby('WELL'):  # pyright: ignore[reportUnknownMemberType]
            assert isinstance(k, str)
            tmp = k.split('*')
            if len(tmp) == 1:
                welldata[k] = {
                        att.name: v.reset_index(drop=True)
                    }
            elif len(tmp) == 2 and not tmp[1]:
                well_names = [name for name in
                    component.main_branches if name.startswith(tmp[0])]
                for name in well_names:
                    welldata[name] = {
                        att.name: v.reset_index(drop=True).assign(WELL=name)
                    }
            else:
                raise ValueError(f'Cound not parse well name "{k}"')
        component.update(welldata, mode='a', ignore_index=True)
        component.fill_na(attr=att.name)
