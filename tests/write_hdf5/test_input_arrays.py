import os
import shutil
import tempfile

import h5py
import numpy as np
import pytest
import torch

from petab_sciml.hdf5.write_hdf5 import write_input_hdf5


@pytest.fixture
def dir_tmp():
    """Create and remove a temporary directory for each test."""
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


def test_write_global_input(dir_tmp):
    """Test writing a global input array under the condition key 0."""
    file1 = os.path.join(dir_tmp, "file1.hdf5")
    input_data = np.random.rand(10)

    result = write_input_hdf5(file1, "input1", input_data)

    assert result == file1
    assert os.path.isfile(file1)

    with h5py.File(file1, "r") as hdf5_file:
        assert "metadata" in hdf5_file
        assert "pytorch_format" in hdf5_file["metadata"]
        assert bool(hdf5_file["metadata"]["pytorch_format"][()]) is True

        assert "inputs" in hdf5_file
        assert "input1" in hdf5_file["inputs"]
        assert "0" in hdf5_file["inputs"]["input1"]

        written_data = hdf5_file["inputs"]["input1"]["0"][()]
        np.testing.assert_array_equal(written_data, input_data)


def test_existing_dataset_raises(dir_tmp):
    """Test that writing an existing dataset raises by default."""
    file1 = os.path.join(dir_tmp, "file1.hdf5")
    input_data = np.random.rand(10)

    write_input_hdf5(file1, "input1", input_data)

    with pytest.raises(ValueError, match="already exists"):
        write_input_hdf5(file1, "input1", input_data)


def test_append_additional_input_to_existing_file(dir_tmp):
    """Test appending a new input dataset to an existing HDF5 file."""
    file1 = os.path.join(dir_tmp, "file1.hdf5")

    input_data1 = np.random.rand(10)
    input_data2 = np.random.rand(10, 10)

    write_input_hdf5(file1, "input1", input_data1)
    write_input_hdf5(
        file1,
        "input2",
        input_data2,
        condition_ids="cond2",
    )

    with h5py.File(file1, "r") as hdf5_file:
        assert bool(hdf5_file["metadata"]["pytorch_format"][()]) is True

        written_data1 = hdf5_file["inputs"]["input1"]["0"][()]
        np.testing.assert_array_equal(written_data1, input_data1)

        assert "input2" in hdf5_file["inputs"]
        assert "cond2" in hdf5_file["inputs"]["input2"]

        written_data2 = hdf5_file["inputs"]["input2"]["cond2"][()]
        np.testing.assert_array_equal(written_data2, input_data2)


def test_overwrite_existing_dataset(dir_tmp):
    """Test overwriting an existing dataset when explicitly requested."""
    file1 = os.path.join(dir_tmp, "file1.hdf5")

    input_data1 = np.random.rand(10, 10)
    input_data2 = np.random.rand(10, 10)

    write_input_hdf5(
        file1,
        "input1",
        input_data1,
        condition_ids="cond1",
    )

    write_input_hdf5(
        file1,
        "input1",
        input_data2,
        condition_ids="cond1",
        on_dataset_exists="overwrite",
    )

    with h5py.File(file1, "r") as hdf5_file:
        written_data = hdf5_file["inputs"]["input1"]["cond1"][()]
        np.testing.assert_array_equal(written_data, input_data2)


def test_write_multiple_conditions(dir_tmp):
    """Test writing condition-specific input arrays for multiple conditions."""
    file2 = os.path.join(dir_tmp, "file2.hdf5")
    input_data = [np.random.rand(10), np.random.rand(10)]

    write_input_hdf5(
        file2,
        "input1",
        input_data,
        condition_ids=["cond1", "cond2"],
    )

    assert os.path.isfile(file2)

    with h5py.File(file2, "r") as hdf5_file:
        assert bool(hdf5_file["metadata"]["pytorch_format"][()]) is True

        assert "inputs" in hdf5_file
        assert "input1" in hdf5_file["inputs"]
        assert "cond1" in hdf5_file["inputs"]["input1"]
        assert "cond2" in hdf5_file["inputs"]["input1"]

        assert "0" not in hdf5_file["inputs"]["input1"]

        written_cond1 = hdf5_file["inputs"]["input1"]["cond1"][()]
        written_cond2 = hdf5_file["inputs"]["input1"]["cond2"][()]

        np.testing.assert_array_equal(written_cond1, input_data[0])
        np.testing.assert_array_equal(written_cond2, input_data[1])


def test_write_torch_tensor_input(dir_tmp):
    """Test writing a PyTorch tensor as an input array."""
    file3 = os.path.join(dir_tmp, "file3.hdf5")
    input_data = torch.rand(10, 10)

    write_input_hdf5(file3, "input1", input_data)

    assert os.path.isfile(file3)

    with h5py.File(file3, "r") as hdf5_file:
        written_data = hdf5_file["inputs"]["input1"]["0"][()]
        expected_data = input_data.detach().cpu().numpy()

        np.testing.assert_array_equal(written_data, expected_data)


def test_write_string_array(dir_tmp):
    """Test that non-numeric input arrays are rejected."""
    file4 = os.path.join(dir_tmp, "file4.hdf5")
    input_data = np.asarray(["1", "2"])
    with pytest.raises(TypeError, match="must be numeric"):
        write_input_hdf5(file4, "input1", input_data)
