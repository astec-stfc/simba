"""Which wake table SIMBA gives Ocelot: the one LAURA wrote for it, whichever extension this LAURA uses."""

import os

from simba.Codes.Ocelot.Ocelot import ocelot_wake_file


def test_the_ocelot_table_is_preferred(tmp_path):
    for extension in (".ocelot", ".astra"):
        (tmp_path / f"wake{extension}").write_text("")
    assert ocelot_wake_file(str(tmp_path), "/lattice/Data_Files/wake.hdf5") == os.path.join(str(tmp_path), "wake.ocelot")


def test_a_laura_that_writes_one_table_for_both_codes_is_still_read(tmp_path):
    (tmp_path / "wake.astra").write_text("")
    assert ocelot_wake_file(str(tmp_path), "wake.hdf5") == os.path.join(str(tmp_path), "wake.astra")


def test_the_file_name_of_a_written_table_gives_the_same_stem(tmp_path):
    (tmp_path / "wake.ocelot").write_text("")
    assert ocelot_wake_file(str(tmp_path), str(tmp_path / "wake.ocelot")) == os.path.join(str(tmp_path), "wake.ocelot")


def test_with_neither_the_astra_name_is_returned(tmp_path):
    assert ocelot_wake_file(str(tmp_path), "wake.hdf5") == os.path.join(str(tmp_path), "wake.astra")
