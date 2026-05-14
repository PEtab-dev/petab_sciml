import os

import pytest
from ruamel.yaml import YAML

from petab_sciml.standard.array_data import add_array_files_to_yaml


def _write_yaml(filename, data):
    """Write test YAML data to file."""
    yaml = YAML()
    with open(filename, "w") as f:
        yaml.dump(data, f)


def _read_yaml(filename):
    """Read test YAML data from file."""
    yaml = YAML()
    with open(filename, "r") as f:
        return yaml.load(f)


def test_add_array_files_to_yaml(dir_tmp):
    """Test adding one and then multiple array files to a PEtab YAML file."""
    yaml_file = os.path.join(dir_tmp, "problem.yaml")
    array_file1 = os.path.join(dir_tmp, "arrays1.hdf5")
    array_file2 = os.path.join(dir_tmp, "arrays2.hdf5")
    array_file3 = os.path.join(dir_tmp, "arrays3.hdf5")

    _write_yaml(
        yaml_file,
        {
            "format_version": 2,
        },
    )

    # Create empty files so paths exist in the YAML directory.
    open(array_file1, "a").close()
    open(array_file2, "a").close()
    open(array_file3, "a").close()

    result = add_array_files_to_yaml(yaml_file, array_file1)

    assert result == yaml_file

    data = _read_yaml(yaml_file)
    assert "extensions" in data
    assert "petab_sciml" in data["extensions"]
    assert data["extensions"]["petab_sciml"]["array_files"] == [
        "arrays1.hdf5",
    ]

    result = add_array_files_to_yaml(
        yaml_file,
        [array_file2, array_file3],
    )

    assert result == yaml_file

    data = _read_yaml(yaml_file)
    assert data["extensions"]["petab_sciml"]["array_files"] == [
        "arrays1.hdf5",
        "arrays2.hdf5",
        "arrays3.hdf5",
    ]


def test_add_existing_array_file_raises(dir_tmp):
    """Test that adding an existing array file raises when requested."""
    yaml_file = os.path.join(dir_tmp, "problem.yaml")
    array_file = os.path.join(dir_tmp, "arrays.hdf5")

    _write_yaml(
        yaml_file,
        {
            "format_version": 2,
            "extensions": {
                "petab_sciml": {
                    "array_files": ["arrays.hdf5"],
                },
            },
        },
    )

    open(array_file, "a").close()

    with pytest.raises(ValueError, match="already listed"):
        add_array_files_to_yaml(
            yaml_file,
            array_file,
            on_existing="raise",
        )


def test_add_array_file_outside_yaml_directory_raises(dir_tmp):
    """Test that array files outside the YAML directory are rejected."""
    yaml_dir = os.path.join(dir_tmp, "yaml_dir")
    array_dir = os.path.join(dir_tmp, "array_dir")

    os.makedirs(yaml_dir)
    os.makedirs(array_dir)

    yaml_file = os.path.join(yaml_dir, "problem.yaml")
    array_file = os.path.join(array_dir, "arrays.hdf5")

    _write_yaml(
        yaml_file,
        {
            "format_version": 2,
        },
    )

    open(array_file, "a").close()

    with pytest.raises(ValueError, match="same directory"):
        add_array_files_to_yaml(yaml_file, array_file)
