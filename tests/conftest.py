from pathlib import Path

import shutil
import tempfile

import pytest


@pytest.fixture
def dir_tmp():
    """Create and remove a temporary directory for each test."""
    directory = Path(tempfile.mkdtemp())
    yield directory
    shutil.rmtree(directory)
