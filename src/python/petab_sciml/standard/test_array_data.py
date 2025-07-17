"""Tests for array data.

TODO move to test suite for python package
"""

import pytest

import numpy as np
from petab_sciml import ArrayData, ArrayDataStandard, METADATA, INPUTS, PARAMETERS, PERM, CONDITION_IDS, ROW, DATA


def test_array_data_dict():
    """Load array data from an dict. Currently no explicit tests..."""
    data = {
        METADATA: {PERM: ROW},
        INPUTS: {
            "inputId1": {
                "0": {
                    CONDITION_IDS: ["cond1", "cond2"],
                    DATA: np.eye(2),
                },
                "1": {
                    CONDITION_IDS: ["cond3", "cond4"],
                    DATA: np.eye(4),
                },
            },
            "inputId2": {"0": {DATA: np.eye(3)}},
        },
    }

    array_data = ArrayDataStandard.model.parse_obj(data)
    # ArrayDataStandard.save_data(array_data, filename="test.hdf5")
