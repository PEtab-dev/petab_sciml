import shutil
import tempfile

import pytest


@pytest.fixture
def dir_tmp():
    """Create and remove a temporary directory for each test."""
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)
