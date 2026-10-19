"""SDDSFile when the SDDS index it asks for is already taken."""

import numpy as np
import pytest

sdds = pytest.importorskip("sdds")

from simba.Modules.SDDSFile import SDDS_Types, SDDSFile  # noqa: E402


def test_a_busy_index_falls_back_to_a_free_one(tmp_path):
    """Asking for an index in use ("Index 1 is already in use") used to leave the
    file with no SDDS object at all, so every read failed; elegant's
    reference-particle trajectory came back empty that way."""
    path = str(tmp_path / "x.sdds")
    written = SDDSFile(index=1)
    written.add_columns(["x"], [np.array([1.5, 2.5])], [SDDS_Types.SDDS_DOUBLE], ["m"], [""])
    written.write_file(path)  # and `written` keeps index 1 busy

    read = SDDSFile(index=1)
    read.read_file(path, page=0)
    assert list(read.columns()["x"].data) == [1.5, 2.5]


def test_column_and_parameter_keep_metadata(tmp_path):
    from simba.Modules.SDDSFile import SDDSColumn, SDDSParameter

    for cls in (SDDSColumn, SDDSParameter):
        obj = cls(name="x", data=1.0, formatstring="%g", fieldlength=8, description="d")
        assert (obj.formatstring, obj.fieldlength, obj.description) == ("%g", 8, "d")
    col = SDDSColumn(name="x", data=[1.0])
    col.fieldlength = 4
    assert col.fieldlength == 4
