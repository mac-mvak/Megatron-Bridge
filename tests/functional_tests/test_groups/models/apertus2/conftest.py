"""Apertus2 tests generate tiny models locally and need no downloaded datasets."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Override the parent fixture's unrelated dataset download."""
