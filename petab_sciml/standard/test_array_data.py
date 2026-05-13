"""Tests for array data.

TODO move to test suite for python package
"""


import numpy as np

from petab_sciml import (
    ALL_CONDITION_IDS,
    INPUTS,
    METADATA,
    PERM,
    ROW,
    ArrayDataStandard,
)


def test_array_data_dict():
    """Load array data from an dict. Currently no explicit tests..."""
    data = {
        METADATA: {PERM: ROW},
        INPUTS: {
            "inputId1": {
                "cond1;cond2": np.eye(2),
                "cond3;cond4": np.eye(4),
            },
            "inputId2": {ALL_CONDITION_IDS: np.eye(3)},
        },
    }

    array_data = ArrayDataStandard.model.parse_obj(data)
    # ArrayDataStandard.save_data(array_data, filename="test.hdf5")
