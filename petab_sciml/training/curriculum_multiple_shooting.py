"""Curriculum multiple shooting export for PEtab problems."""

import copy
from pathlib import Path

import petab.v2

from .partition import get_partition_time_points
from .multiple_shooting import _get_ms_problem


def _export_curriculum_multiple_shooting(
    problem: petab.v2.Problem,
    strategy,  # CurriculumMultipleShooting
    output_dir: Path,
    validate: bool,
) -> None:
    """Export a PEtab problem as a curriculum multiple shooting training problem.

    Each stage is itself a multiple shooting problem with progressively fewer
    windows. The first stage uses the full set of windows from the partition;
    at each subsequent stage, the last window is dropped and the remaining
    windows extend to cover the full time range.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ms_strategy = strategy.multiple_shooting
    measurement_df = problem.measurement_df
    end_times = get_partition_time_points(ms_strategy.partition, measurement_df)
    t_min = float(measurement_df["time"].min())
    starts = [t_min] + end_times[:-1]
    n_stages = len(end_times)

    for stage_idx in range(n_stages):
        n_windows = n_stages - stage_idx
        stage_windows = list(zip(starts[:n_windows], end_times[stage_idx:]))

        # Last stage corresponds to the original problem
        if stage_idx == n_stages - 1:
            stage_problem = copy.deepcopy(problem)
        else:
            stage_problem = _get_ms_problem(problem, stage_windows, ms_strategy)

        if validate:
            stage_problem.validate()

        stage_dir = output_dir / f"stage{stage_idx + 1}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_problem.to_files(stage_dir)
