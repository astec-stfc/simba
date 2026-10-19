"""Shared fixtures."""

import pytest
from helpers import fodo_quads


@pytest.fixture
def fodo_elements():
    """A fresh QUAD1F / QUAD1D pair."""
    return fodo_quads()
