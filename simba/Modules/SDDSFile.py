"""SDDS file access, now kept in LAURA. Each :class:`SDDSFile` has its own SDDS slot."""

from laura.translator.utils.elegant.sdds_file import SDDSFile  # noqa: F401
from laura.translator.utils.sdds_file import (  # noqa: F401
    SDDSArray,
    SDDSColumn,
    SDDSObject,
    SDDSParameter,
    SddsTypes,
    read_sdds_file,
)

SDDS_Types = SddsTypes
