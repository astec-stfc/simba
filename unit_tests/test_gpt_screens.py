"""GPT screens the beam never reached: post-processing skips them rather than crashing."""

from types import SimpleNamespace

import numpy as np
import pytest

from simba.Modules import Beams as rbf


class _Gdf:
    """GPT output with one screen: an empty block (no particle crossed it) or none at all."""

    def __init__(self, block):
        self.block = block

    def get_position(self, position):
        return self.block


@pytest.mark.parametrize(
    "block, message",
    [(SimpleNamespace(x=np.array([])), "no particles"), (None, "DID NOT find")],
)
def test_an_empty_or_missing_screen_is_a_value_error(block, message):
    # gptLattice.gdf_to_hdf5 turns a ValueError into a warning and skips the screen
    with pytest.raises(ValueError, match=message):
        rbf.gdf.read_gdf_beam_file(rbf.beam(), position=-0.5, gdfbeam=_Gdf(block))
