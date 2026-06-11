"""Tests for multiple shooting export on a problem without experiments."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import numpy as np
from petab.v1.math import sympify_petab

from petab_sciml.training import (
    CustomPartition,
    MultipleShootingProblem,
    UniformPartition,
)

from tests.training.petab_problems import (
    get_prob_no_experiment,
    get_prob_partial_experiments_ms,
)
from tests.training.ms_helpers import (
    _assert_ms_structure,
    _assert_penalty_measurements,
    _assert_window_measurements_match,
)


# -----------------------------------------------------------------------------
# Problem 1: No PEtab experiments
# -----------------------------------------------------------------------------
def test_multiple_shooting_no_experiments(dir_tmp: Path) -> None:
    """Test multiple shooting export on a problem without experiments.

    Uses hard-coded expected DataFrames for the UniformPartition case to verify
    exact correctness, and invariant checks for the CustomPartition case to
    verify structural correctness without coupling to row-level details.
    """
    path_yaml = dir_tmp / "problem.yaml"
    get_prob_no_experiment(dir_tmp)
    species = ["x1", "x2"]

    # UniformPartition(n=3): hard-coded reference test to ensure all formulas
    # are correct
    dir_uniform = dir_tmp / "dir_uniform"
    MultipleShootingProblem(
        yaml=path_yaml, partition=UniformPartition(n=3), penalty=5.0
    ).export(dir_uniform)

    # Expected parameter table
    expected_params = pd.DataFrame(
        [
            {
                "parameterId": "k1",
                "lowerBound": 1e-3,
                "upperBound": 1e3,
                "estimate": True,
            },
            {
                "parameterId": "k2",
                "lowerBound": 1e-3,
                "upperBound": 1e3,
                "estimate": True,
            },
            {
                "parameterId": "MS_PENALTY_SQRT",
                "nominalValue": math.sqrt(5.0),
                "estimate": False,
            },
            {
                "parameterId": "WINDOW1_EXPERIMENT_default_experiment_PARAMETER_x1",
                "lowerBound": 1e-8,
                "upperBound": 1e8,
                "nominalValue": 1e-6,
                "estimate": True,
            },
            {
                "parameterId": "WINDOW1_EXPERIMENT_default_experiment_PARAMETER_x2",
                "lowerBound": 1e-8,
                "upperBound": 1e8,
                "nominalValue": 1e-6,
                "estimate": True,
            },
            {
                "parameterId": "WINDOW2_EXPERIMENT_default_experiment_PARAMETER_x1",
                "lowerBound": 1e-8,
                "upperBound": 1e8,
                "nominalValue": 1e-6,
                "estimate": True,
            },
            {
                "parameterId": "WINDOW2_EXPERIMENT_default_experiment_PARAMETER_x2",
                "lowerBound": 1e-8,
                "upperBound": 1e8,
                "nominalValue": 1e-6,
                "estimate": True,
            },
        ]
    )
    _assert_table_equal(
        expected_params, dir_uniform / "parameters.tsv", key="parameterId"
    )

    # Expected condition table
    expected_conditions = pd.DataFrame(
        [
            {
                "conditionId": "WINDOW1_EXPERIMENT_default_experiment_IC",
                "targetId": "x1",
                "targetValue": "WINDOW1_EXPERIMENT_default_experiment_PARAMETER_x1",
            },
            {
                "conditionId": "WINDOW1_EXPERIMENT_default_experiment_IC",
                "targetId": "x2",
                "targetValue": "WINDOW1_EXPERIMENT_default_experiment_PARAMETER_x2",
            },
            {
                "conditionId": "WINDOW2_EXPERIMENT_default_experiment_IC",
                "targetId": "x1",
                "targetValue": "WINDOW2_EXPERIMENT_default_experiment_PARAMETER_x1",
            },
            {
                "conditionId": "WINDOW2_EXPERIMENT_default_experiment_IC",
                "targetId": "x2",
                "targetValue": "WINDOW2_EXPERIMENT_default_experiment_PARAMETER_x2",
            },
        ]
    )
    _assert_table_equal(
        expected_conditions,
        dir_uniform / "conditions.tsv",
        key=["conditionId", "targetId"],
    )

    # Expected experiment table
    expected_experiments = pd.DataFrame(
        [
            {
                "experimentId": "WINDOW0_EXPERIMENT_default_experiment",
                "time": 0.0,
                "conditionId": np.nan,
            },
            {
                "experimentId": "WINDOW1_EXPERIMENT_default_experiment",
                "time": 1.0,
                "conditionId": "WINDOW1_EXPERIMENT_default_experiment_IC",
            },
            {
                "experimentId": "WINDOW2_EXPERIMENT_default_experiment",
                "time": 3.0,
                "conditionId": "WINDOW2_EXPERIMENT_default_experiment_IC",
            },
        ]
    )
    _assert_table_equal(
        expected_experiments,
        dir_uniform / "experiments.tsv",
        key=["experimentId", "time"],
    )

    # Expected measurement table
    rows = []
    for t in [0.0, 1.0]:
        rows.append(
            {
                "observableId": "obs1",
                "experimentId": "WINDOW0_EXPERIMENT_default_experiment",
                "time": t,
                "measurement": t * 0.5,
            }
        )
        rows.append(
            {
                "observableId": "obs2",
                "experimentId": "WINDOW0_EXPERIMENT_default_experiment",
                "time": t,
                "measurement": t * 0.3,
            }
        )
    for specie in species:
        rows.append(
            {
                "observableId": f"WINDOW1_EXPERIMENT_default_experiment_PENALTY_{specie}",
                "experimentId": "WINDOW0_EXPERIMENT_default_experiment",
                "time": 1.0,
                "measurement": 0.0,
            }
        )
    for t in [1.0, 2.0, 3.0]:
        rows.append(
            {
                "observableId": "obs1",
                "experimentId": "WINDOW1_EXPERIMENT_default_experiment",
                "time": t,
                "measurement": t * 0.5,
            }
        )
        rows.append(
            {
                "observableId": "obs2",
                "experimentId": "WINDOW1_EXPERIMENT_default_experiment",
                "time": t,
                "measurement": t * 0.3,
            }
        )
    for specie in species:
        rows.append(
            {
                "observableId": f"WINDOW2_EXPERIMENT_default_experiment_PENALTY_{specie}",
                "experimentId": "WINDOW1_EXPERIMENT_default_experiment",
                "time": 3.0,
                "measurement": 0.0,
            }
        )
    for t in [3.0, 4.0, 5.0, 6.0]:
        rows.append(
            {
                "observableId": "obs1",
                "experimentId": "WINDOW2_EXPERIMENT_default_experiment",
                "time": t,
                "measurement": t * 0.5,
            }
        )
        rows.append(
            {
                "observableId": "obs2",
                "experimentId": "WINDOW2_EXPERIMENT_default_experiment",
                "time": t,
                "measurement": t * 0.3,
            }
        )
    expected_measurements = pd.DataFrame(rows)
    _assert_table_equal(
        expected_measurements,
        dir_uniform / "measurements.tsv",
        key=["experimentId", "observableId", "time", "measurement"],
    )

    # Observable table: penalty observable formulas need sympy-aware comparison,
    # so we check them separately rather than via _assert_table_equal
    observables = pd.read_csv(dir_uniform / "observables.tsv", sep="\t")
    assert set(observables["observableId"]) == {
        "obs1",
        "obs2",
        "WINDOW1_EXPERIMENT_default_experiment_PENALTY_x1",
        "WINDOW1_EXPERIMENT_default_experiment_PENALTY_x2",
        "WINDOW2_EXPERIMENT_default_experiment_PENALTY_x1",
        "WINDOW2_EXPERIMENT_default_experiment_PENALTY_x2",
    }
    for window_idx in (1, 2):
        for specie in species:
            obs_id = (
                f"WINDOW{window_idx}_EXPERIMENT_default_experiment_PENALTY_{specie}"
            )
            param_id = (
                f"WINDOW{window_idx}_EXPERIMENT_default_experiment_PARAMETER_{specie}"
            )
            row = observables[observables["observableId"] == obs_id].iloc[0]
            assert sympify_petab(row["observableFormula"]) == sympify_petab(
                f"({specie} - {param_id}) * MS_PENALTY_SQRT"
            )
            assert sympify_petab(row["noiseFormula"]) == sympify_petab("1")
            assert row["noiseDistribution"] == "normal"

    # CustomPartition([2.5, 5.0]): invariant-based test
    # End times [2.5, 5.0, 6.0], windows [0,2.5], [2.5,5.0], [5.0,6.0]
    dir_custom = dir_tmp / "dir_custom"
    ms_prob = MultipleShootingProblem(
        yaml=path_yaml, partition=CustomPartition([2.5, 5.0]), penalty=4.0
    )
    ms_prob.export(dir_custom)

    windows = [(0.0, 2.5), (2.5, 5.0), (5.0, 6.0)]
    _assert_ms_structure(
        dir_custom,
        windows=windows,
        species=species,
        original_yaml=path_yaml,
        prob=ms_prob,
    )

    _assert_window_measurements_match(dir_custom, path_yaml, windows)
    _assert_penalty_measurements(
        dir_custom, windows=windows, species=species, original_yaml=path_yaml
    )


# -----------------------------------------------------------------------------
# Problem 2: PEtab experiments with unequal measurements
# -----------------------------------------------------------------------------
def test_multiple_shooting_partial_experiments(dir_tmp: Path) -> None:
    """Problem where one of two experiments lacks measurements in the last window."""
    path_yaml = dir_tmp / "problem.yaml"
    get_prob_partial_experiments_ms(dir_tmp)
    species = ["x1", "x2"]

    # UniformPartition(n=3) → end times [1.0, 3.0, 6.0], windows [0,1], [1,3], [3,6]
    # exp1 participates in all three windows; exp2 only in windows 0 and 1
    dir_uniform = dir_tmp / "dir_uniform"
    ms_prob = MultipleShootingProblem(
        yaml=path_yaml, partition=UniformPartition(n=3), penalty=5.0, log_penalty=True
    )
    ms_prob.export(dir_uniform)

    windows = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0)]
    _assert_ms_structure(
        dir_uniform,
        windows=windows,
        species=species,
        original_yaml=path_yaml,
        prob=ms_prob,
    )
    _assert_window_measurements_match(dir_uniform, path_yaml, windows)
    _assert_penalty_measurements(
        dir_uniform,
        windows=windows,
        species=species,
        original_yaml=path_yaml,
    )


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _assert_table_equal(
    expected: pd.DataFrame,
    actual_path: Path,
    key,
) -> None:
    """Compare expected DataFrame against TSV at actual_path, sorted by key columns.

    Only columns present in ``expected`` are compared; the actual TSV may have
    additional columns. Both sides are sorted by ``key`` to remove row-order
    fragility.
    """
    actual = pd.read_csv(actual_path, sep="\t")
    sort_cols = [key] if isinstance(key, str) else list(key)
    expected = expected.sort_values(sort_cols).reset_index(drop=True)
    actual = actual.sort_values(sort_cols).reset_index(drop=True)
    actual = actual[expected.columns]
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
