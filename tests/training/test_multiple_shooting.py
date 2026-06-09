"""Tests for multiple shooting export on a problem without experiments."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import petab.v2
import numpy as np
import pytest
from petab.v1.math import sympify_petab

from petab_sciml.training.partition import CustomPartition, UniformPartition
from petab_sciml.training.strategies import (
    MultipleShooting,
    PEtabTrainingProblem,
)

from tests.training.petab_problems import (
    get_prob_no_experiment,
    get_prob_partial_experiments_ms,
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
    PEtabTrainingProblem(
        yaml=path_yaml,
        strategy=MultipleShooting(UniformPartition(n=3), penalty=5.0),
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
    strategy = MultipleShooting(CustomPartition([2.5, 5.0]), penalty=4.0)
    PEtabTrainingProblem(yaml=path_yaml, strategy=strategy).export(dir_custom)

    windows = [(0.0, 2.5), (2.5, 5.0), (5.0, 6.0)]
    _assert_ms_structure(
        dir_custom,
        windows=windows,
        species=species,
        original_yaml=path_yaml,
        strategy=strategy,
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
    strategy = MultipleShooting(UniformPartition(n=3), penalty=5.0, log_penalty=True)
    PEtabTrainingProblem(yaml=path_yaml, strategy=strategy).export(dir_uniform)

    windows = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0)]
    _assert_ms_structure(
        dir_uniform,
        windows=windows,
        species=species,
        original_yaml=path_yaml,
        strategy=strategy,
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
def _compute_exp_windows(
    original_yaml: Path,
    windows: list[tuple[float, float]],
) -> dict[str, list[int]]:
    """Compute which window indices each (original) experiment participates in"""
    original = petab.v2.Problem.from_yaml(original_yaml)
    if original.experiments:
        t_end_per_exp = {
            exp.id: max(
                (m.time for m in original.measurements if m.experiment_id == exp.id),
                default=float("-inf"),
            )
            for exp in original.experiments
        }
    else:
        t_end_per_exp = {"default_experiment": original.measurement_df["time"].max()}

    return {
        exp_id: [i for i, (t0, _) in enumerate(windows) if t_end >= t0]
        for exp_id, t_end in t_end_per_exp.items()
    }


def _assert_ms_structure(
    problem_dir: Path,
    windows: list[tuple[float, float]],
    species: list[str],
    original_yaml: Path,
    strategy: MultipleShooting,
) -> None:
    """MS-specific structure: per-(window i>0, experiment)"""
    exp_windows = _compute_exp_windows(original_yaml, windows)

    parameters = pd.read_csv(problem_dir / "parameters.tsv", sep="\t")
    observables = pd.read_csv(problem_dir / "observables.tsv", sep="\t")
    conditions = pd.read_csv(problem_dir / "conditions.tsv", sep="\t")
    experiments = pd.read_csv(problem_dir / "experiments.tsv", sep="\t")

    ms_row = parameters[parameters["parameterId"] == "MS_PENALTY_SQRT"].iloc[0]
    assert ms_row["estimate"] == np.False_
    assert float(ms_row["nominalValue"]) == math.sqrt(strategy.penalty)

    param_ids = set(parameters["parameterId"])
    obs_ids = set(observables["observableId"])
    cond_ids = set(conditions["conditionId"])
    exp_ids = set(experiments["experimentId"])

    for exp_id, window_indices in exp_windows.items():
        for i in window_indices:
            assert f"WINDOW{i}_EXPERIMENT_{exp_id}" in exp_ids
            if i == 0:
                continue
            prefix = f"WINDOW{i}_EXPERIMENT_{exp_id}"
            for s in species:
                pid = f"{prefix}_PARAMETER_{s}"
                assert pid in param_ids, f"Missing parameter {pid}"
                p = parameters[parameters["parameterId"] == pid].iloc[0]
                assert p["estimate"] == np.True_
                assert float(p["nominalValue"]) == strategy.initial_value
                assert f"{prefix}_PENALTY_{s}" in obs_ids
            assert f"{prefix}_IC" in cond_ids


def _assert_penalty_measurements(
    problem_dir: Path,
    windows: list[tuple[float, float]],
    species: list[str],
    original_yaml: Path,
) -> None:
    """Penalty measurements tests"""
    exp_windows = _compute_exp_windows(original_yaml, windows)

    exported_meas = pd.read_csv(problem_dir / "measurements.tsv", sep="\t")
    penalty_meas = exported_meas[exported_meas["observableId"].str.contains("PENALTY")]
    assert (penalty_meas["measurement"] == 0.0).all()

    expected_count = 0
    expected_times = set()
    for window_indices in exp_windows.values():
        for i in window_indices:
            if (i + 1) in window_indices:
                expected_count += len(species)
                expected_times.add(windows[i][1])

    assert set(penalty_meas["time"]) == expected_times
    assert len(penalty_meas) == expected_count


def _assert_window_measurements_match(
    problem_dir: Path,
    original_yaml: Path,
    windows: list[tuple[float, float]],
) -> None:
    """Per-window non-penalty measurements equal original measurements filtered to [t0, tf]."""
    original_meas = petab.v2.Problem.from_yaml(original_yaml).measurement_df
    exported_meas = pd.read_csv(problem_dir / "measurements.tsv", sep="\t")
    obs_meas = exported_meas[~exported_meas["observableId"].str.contains("PENALTY")]

    for i, (t0, tf) in enumerate(windows):
        window_meas = obs_meas[
            obs_meas["experimentId"].str.startswith(f"WINDOW{i}_EXPERIMENT_")
        ]
        original_in_range = original_meas[
            (original_meas["time"] >= t0) & (original_meas["time"] <= tf)
        ]
        assert len(window_meas) == len(original_in_range), (
            f"Window {i} measurement count mismatch: "
            f"{len(window_meas)} vs {len(original_in_range)}"
        )
        assert sorted(window_meas["measurement"]) == sorted(
            original_in_range["measurement"]
        )


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
