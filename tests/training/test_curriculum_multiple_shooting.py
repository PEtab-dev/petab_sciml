"""Tests for curriculum multiple shooting export."""

from pathlib import Path

from petab_sciml.training.partition import CustomPartition, UniformPartition
from petab_sciml.training.strategies import (
    CurriculumMultipleShooting,
    MultipleShooting,
    PEtabTrainingProblem,
)
from tests.training.petab_problems import (
    get_prob_no_experiment,
    get_prob_partial_experiments_ms
)
from tests.training.ms_helpers import (
    _assert_ms_structure,
    _assert_penalty_measurements,
    _assert_window_measurements_match,
)


def test_curriculum_multiple_shooting_no_experiments(dir_tmp: Path) -> None:
    """Test CMS export on a problem without experiments"""
    path_yaml = dir_tmp / "problem.yaml"
    get_prob_no_experiment(dir_tmp)
    species = ["x1", "x2"]

    # UniformPartition(n=3) → end times [1.0, 3.0, 6.0]
    dir_uniform = dir_tmp / "dir_uniform"
    strategy_uniform = CurriculumMultipleShooting(
        multiple_shooting=MultipleShooting(UniformPartition(n=3), penalty=5.0)
    )
    PEtabTrainingProblem(yaml=path_yaml, strategy=strategy_uniform).export(dir_uniform)

    stage_windows_uniform = [
        [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0)],
        [(0.0, 3.0), (1.0, 6.0)],
        [(0.0, 6.0)],
    ]
    _check_cms_stages(
        dir_uniform,
        path_yaml,
        stage_windows_uniform,
        species,
        strategy_uniform.multiple_shooting,
    )

    # CustomPartition([2.5, 5.0]) → end times [2.5, 5.0, 6.0]
    dir_custom = dir_tmp / "dir_custom"
    strategy_custom = CurriculumMultipleShooting(
        multiple_shooting=MultipleShooting(CustomPartition([2.5, 5.0]), penalty=5.0)
    )
    PEtabTrainingProblem(yaml=path_yaml, strategy=strategy_custom).export(dir_custom)

    stage_windows_custom = [
        [(0.0, 2.5), (2.5, 5.0), (5.0, 6.0)],
        [(0.0, 5.0), (2.5, 6.0)],
        [(0.0, 6.0)],
    ]
    _check_cms_stages(
        dir_custom,
        path_yaml,
        stage_windows_custom,
        species,
        strategy_custom.multiple_shooting,
    )


def test_curriculum_multiple_shooting_partial_experiments(dir_tmp: Path) -> None:
    """Export on a problem where one experiments lacks measurements in the last window"""
    path_yaml = dir_tmp / "problem.yaml"
    get_prob_partial_experiments_ms(dir_tmp)
    species = ["x1", "x2"]

    dir_out = dir_tmp / "dir_cms"
    strategy = CurriculumMultipleShooting(
        multiple_shooting=MultipleShooting(UniformPartition(n=3), penalty=5.0)
    )
    PEtabTrainingProblem(yaml=path_yaml, strategy=strategy).export(dir_out)

    # UniformPartition(n=3) on 7 time points → end times [1.0, 3.0, 6.0]
    # Stage 1: [(0,1), (1,3), (3,6)]   — exp1 in all, exp2 only in [0] and [1]
    # Stage 2: [(0,3), (1,6)]          — exp1 in both, exp2 in [0] only (max=2 < 1? no, in both)
    # Stage 3: [(0,6)]                 — single window (= original)
    stage_windows = [
        [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0)],
        [(0.0, 3.0), (1.0, 6.0)],
        [(0.0, 6.0)],
    ]
    _check_cms_stages(
        dir_out,
        path_yaml,
        stage_windows,
        species,
        strategy.multiple_shooting,
    )


def _check_cms_stages(
    dir_out: Path,
    path_yaml: Path,
    stage_windows: list[list[tuple[float, float]]],
    species: list[str],
    ms_strategy: MultipleShooting,
) -> None:
    """Run the standard MS invariant checks on each stage directory.

    Skips the final stage, which corresponds to the original single-window
    problem.
    """
    for stage_idx, windows in enumerate(stage_windows):
        if stage_idx + 1 == len(stage_windows):
            continue

        stage_dir = dir_out / f"stage{stage_idx + 1}"
        _assert_ms_structure(
            stage_dir,
            windows=windows,
            species=species,
            original_yaml=path_yaml,
            strategy=ms_strategy,
        )
        _assert_window_measurements_match(stage_dir, path_yaml, windows)
        _assert_penalty_measurements(
            stage_dir,
            windows=windows,
            species=species,
            original_yaml=path_yaml,
        )
