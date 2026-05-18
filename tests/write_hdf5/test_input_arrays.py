import os
import pytest

import h5py
import numpy as np
from pydantic import ValidationError

from petab_sciml.standard.array_data import (
    ArrayData,
    ArrayDataStandard,
    ALL_CONDITION_IDS,
)


def test_array_data_requires_pytorch_format_metadata():
    """Test validation works"""
    data = {
        "metadata": {"pytorch": True},
        "inputs": {
            "inputId2": {
                ALL_CONDITION_IDS: np.eye(3),
            },
        },
    }

    with pytest.raises(ValidationError, match="pytorch_format"):
        ArrayData.model_validate(data)


def test_array_data_input_data(dir_tmp):
    """Test writing input array data"""
    input1 = np.random.rand(10)
    input2 = np.random.rand(10, 10)
    input3 = np.eye(3)

    data = {
        "metadata": {"pytorch_format": True},
        "inputs": {
            "inputId1": {
                "cond1;cond2": input1,
                "cond3;cond4": input2,
            },
            "inputId2": {
                ALL_CONDITION_IDS: input3,
            },
        },
    }

    array_data = ArrayData.model_validate(data)

    filename = os.path.join(dir_tmp, "array_data.hdf5")
    ArrayDataStandard.save_data(array_data, filename=filename)

    assert os.path.isfile(filename)

    with h5py.File(filename, "r") as hdf5_file:
        # Metadata
        assert "metadata" in hdf5_file
        assert "pytorch_format" in hdf5_file["metadata"]
        assert bool(hdf5_file["metadata"]["pytorch_format"][()]) is True

        # Top-level input group
        assert "inputs" in hdf5_file

        # inputId1: condition-specific arrays
        assert "inputId1" in hdf5_file["inputs"]
        input_group1 = hdf5_file["inputs"]["inputId1"]

        assert "cond1;cond2" in input_group1
        assert "cond3;cond4" in input_group1

        np.testing.assert_array_equal(
            input_group1["cond1;cond2"][()],
            input1,
        )
        np.testing.assert_array_equal(
            input_group1["cond3;cond4"][()],
            input2,
        )

        # inputId2: global input array
        assert "inputId2" in hdf5_file["inputs"]
        input_group2 = hdf5_file["inputs"]["inputId2"]

        assert ALL_CONDITION_IDS in input_group2
        np.testing.assert_array_equal(
            input_group2[ALL_CONDITION_IDS][()],
            input3,
        )
