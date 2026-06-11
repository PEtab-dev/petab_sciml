import pytest

import numpy as np
import pandas as pd
from petab_sciml.training import (
    CustomPartition,
    UniformPartition,
)

# DataFrames for testing partitioning strategies
basic_df = pd.DataFrame(
    {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "observableId": ["A"] * 7 + ["B"] * 7,
        "measurement": np.random.rand(14),
    }
)
unequal_measurements_df = pd.DataFrame(
    {
        "time": [0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0],
        "observableId": ["A", "A", "A", "B", "C", "A", "A"],
        "measurement": [1.0, 2.0, 3.0, 3.1, 2.9, 4.0, 3.0],
    }
)
exceeds_t_end_df = pd.DataFrame(
    {
        "time": [0.0, 1.0, 2.0, 3.0],
        "observableId": ["A"] * 4,
        "measurement": [1.0, 2.0, 3.0, 2.0],
    }
)
gap_in_data_df = pd.DataFrame(
    {
        "time": [0.0, 1.0, 3.0, 4.0, 5.0],
        "observableId": ["A"] * 5,
        "measurement": [1.0, 2.0, 3.0, 2.0, 1.0],
    }
)
too_few_timepoints_df = pd.DataFrame(
    {
        "time": [0.0, 1.0, 2.0],
        "observableId": ["A"] * 3,
        "measurement": [1.0, 2.0, 1.0],
    }
)


def test_partition_strategies():
    """Test that uniform and custom partitions return correct end time points."""
    uniform1 = UniformPartition(n=3).get_time_points(basic_df)
    uniform2 = UniformPartition(n=7).get_time_points(basic_df)
    custom1 = CustomPartition(interior_points=[2.0, 4.0]).get_time_points(basic_df)
    custom2 = CustomPartition(interior_points=[2.5, 4.5]).get_time_points(basic_df)
    assert uniform1 == [1.0, 3.0, 6.0]
    assert uniform2 == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert custom1 == [2.0, 4.0, 6.0]
    assert custom2 == [2.5, 4.5, 6.0]

    # Unequal measurements, test splitting is on unique time-points
    uniform1 = UniformPartition(3).get_time_points(unequal_measurements_df)
    assert uniform1 == [0.0, 2.0, 4.0]


def test_input_validation():
    """Test that invalid partition configurations raise descriptive errors."""
    with pytest.raises(ValueError, match="interior_points must be strictly less"):
        CustomPartition([1.0, 5.0]).get_time_points(exceeds_t_end_df)

    with pytest.raises(ValueError, match="contains no data points"):
        CustomPartition([1.5, 2.5]).get_time_points(gap_in_data_df)

    with pytest.raises(ValueError, match="Cannot create 4 segments"):
        UniformPartition(4).get_time_points(too_few_timepoints_df)

    with pytest.raises(ValueError, match="n must be >= 2, got 1."):
        UniformPartition(n=1)

    with pytest.raises(ValueError, match="interior_points must be strictly increasing"):
        CustomPartition([2.0, 1.0])
