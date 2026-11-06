"""``mr`` runs gpt through cmd without quoting its path, so the crest scan needs one with no spaces."""

import os

import pytest

from simba.Codes.GPT.GPT import _without_spaces


def test_a_path_without_spaces_is_left_alone():
    assert _without_spaces("/opt/gpt/bin/gpt") == "/opt/gpt/bin/gpt"


@pytest.mark.skipif(os.name != "nt", reason="8.3 names are a Windows thing")
def test_a_windows_path_with_spaces_gets_its_short_name(tmp_path):
    folder = tmp_path / "Program Files"
    folder.mkdir()
    (folder / "gpt.exe").write_text("")
    assert " " not in _without_spaces(str(folder / "gpt.exe"))
