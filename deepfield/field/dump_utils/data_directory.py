from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

class DataTypes(Enum):
    STRING = auto()
    DATE = auto()


@dataclass
class DirectoryEntrySpecification:
    keyword: str
    data_type: Optional[DataTypes]
    getter: Optional[Callable]
    is_present: Callable = lambda _: True

DATA_DIRECTORY = {
    "RUNSPEC": [
        DirectoryEntrySpecification('TITLE', DataTypes.STRING, lambda field: field.meta['TITLE']),
        DirectoryEntrySpecification('MULTOUT', None, None),
        DirectoryEntrySpecification('MULTOUTS', None, None),
        DirectoryEntrySpecification('START', DataTypes.STRING, lambda field: field.meta['START']),
        DirectoryEntrySpecification('METRIC', None, None, lambda field: field.meta['UNITS'] == 'METRIC'),
        *[
            DirectoryEntrySpecification(
                fluid, None, None, lambda field: fluid in field.meta['FLUIDS']
            ) for fluid in ('OIL', 'GAS', 'WATER')
        ]
    ]
}

_DUMP_ROUTINES = {
    DataTypes.STRING: lambda val: val,
}

def dump_keyword(spec, val):
    return "\n".join([spec.keyword] + ([_DUMP_ROUTINES[spec.data_type](val)] if spec.data_type else []))

