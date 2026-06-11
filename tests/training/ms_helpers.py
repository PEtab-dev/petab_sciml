import math
from pathlib import Path

import numpy as np
import pandas as pd
import petab

from petab_sciml.training import (
    CurriculumMultipleShootingProblem,
    MultipleShootingProblem,
)


def _compute_exp_windows(
    original_yaml: Path,
    windows: list[tuple[float, float]],
) -> dict[str, list[int]]:
    """Compute which window indices each original experiment participates in.

    An experiment participates in window i iff max(its measurement times) >= t0_i.
    For problems without an experiment table, MS export adds 'default_experiment'.
    """
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
    prob: MultipleShootingProblem | CurriculumMultipleShootingProblem,
) -> None:
    """MS-specific structure: per-(window i>0, experiment)."""
    exp_windows = _compute_exp_windows(original_yaml, windows)

    parameters = pd.read_csv(problem_dir / "parameters.tsv", sep="\t")
    observables = pd.read_csv(problem_dir / "observables.tsv", sep="\t")
    conditions = pd.read_csv(problem_dir / "conditions.tsv", sep="\t")
    experiments = pd.read_csv(problem_dir / "experiments.tsv", sep="\t")

    ms_row = parameters[parameters["parameterId"] == "MS_PENALTY_SQRT"].iloc[0]
    assert ms_row["estimate"] == np.False_
    assert float(ms_row["nominalValue"]) == math.sqrt(prob.penalty)

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
                assert float(p["nominalValue"]) == prob.initial_value
                assert f"{prefix}_PENALTY_{s}" in obs_ids
            assert f"{prefix}_IC" in cond_ids


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


def _assert_penalty_measurements(
    problem_dir: Path,
    windows: list[tuple[float, float]],
    species: list[str],
    original_yaml: Path,
) -> None:
    """Penalty measurements: value 0, placed at t0 of the next window when the experiment continues."""
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
                expected_times.add(windows[i + 1][0])

    assert set(penalty_meas["time"]) == expected_times
    assert len(penalty_meas) == expected_count
