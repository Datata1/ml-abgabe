"""pytest-Konfiguration.

Tests, die die lokalen TEP-Parquet-Dateien benötigen, sind mit ``@pytest.mark.data`` markiert
und werden **automatisch übersprungen**, wenn die Dateien fehlen (``pytest -m "not data"``).
"""

from __future__ import annotations

import pytest

from automl_ad import config

_DATA_FILES = [
    config.FAULT_FREE_TRAINING,
    config.FAULT_FREE_TESTING,
    config.FAULTY_TESTING,
]


def data_available() -> bool:
    return all(p.exists() for p in _DATA_FILES)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 - pytest-Hook-Signatur
    if data_available():
        return
    skip_data = pytest.mark.skip(reason="TEP-Parquet-Dateien fehlen unter data/")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip_data)
