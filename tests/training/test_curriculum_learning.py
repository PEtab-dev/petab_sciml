from pathlib import Path

import pandas as pd
import petab.v2

from petab_sciml.training.partition import CustomPartition, UniformPartition
from petab_sciml.training.strategies import CurriculumLearning, PEtabTrainingProblem

from tests.training.petab_problems import (
    get_prob_no_experiment,
    get_prob_all_experiments,
    get_prob_partial_experiments,
)


# -----------------------------------------------------------------------------
# Problem 1: No PEtab experiments
# -----------------------------------------------------------------------------
def test_no_experiments(dir_tmp: Path) -> None:
    """Test uniform and custom partitions on a problem without experiments."""
    path_yaml = dir_tmp / "problem.yaml"
    get_prob_no_experiment(dir_tmp)

    dir_uniform = dir_tmp / "dir_uniform"
    training_problem1 = PEtabTrainingProblem(
        yaml=path_yaml, strategy=CurriculumLearning(UniformPartition(n=3))
    )
    training_problem1.export(dir_uniform)
    end_times = [1.0, 3.0, 6.0]
    for i, t_end in enumerate(end_times):
        measurements_path = dir_uniform / f"stage{i + 1}" / "measurements.tsv"
        measurements_df = pd.read_csv(measurements_path, delimiter="\t")
        assert all(measurements_df["time"] <= t_end)
        assert any(measurements_df["time"] == t_end)

    # Also tests setting partition points at no measurements is handled
    # correctly
    dir_custom = dir_tmp / "dir_custom"
    training_problem2 = PEtabTrainingProblem(
        yaml=path_yaml, strategy=CurriculumLearning(CustomPartition([2.5, 5.0]))
    )
    training_problem2.export(dir_custom)
    end_times = [2.0, 5.0, 6.0]
    for i, t_end in enumerate(end_times):
        measurements_path = dir_custom / f"stage{i + 1}" / "measurements.tsv"
        measurements_df = pd.read_csv(measurements_path, delimiter="\t")
        assert all(measurements_df["time"] <= t_end)
        assert any(measurements_df["time"] == t_end)


# -----------------------------------------------------------------------------
# Problem 2: With experiments, all experiments cover all time points
# -----------------------------------------------------------------------------
def test_all_experiments(dir_tmp: Path) -> None:
    path_yaml = dir_tmp / "problem.yaml"
    get_prob_all_experiments(dir_tmp)

    dir_uniform = dir_tmp / "dir_uniform"
    training_problem1 = PEtabTrainingProblem(
        yaml=path_yaml, strategy=CurriculumLearning(UniformPartition(n=3))
    )
    training_problem1.export(dir_uniform)
    end_times = [1.0, 3.0, 6.0]

    original_problem = petab.v2.Problem.from_yaml(path_yaml)
    for i, t_end in enumerate(end_times):
        stage = petab.v2.Problem.from_yaml(
            dir_uniform / f"stage{i + 1}" / "problem.yaml"
        )
        # All experiments survive since both cover all time points
        assert {e.id for e in stage.experiments} == {
            e.id for e in original_problem.experiments
        }
        # All conditions survive since all experiments survive
        assert {c.id for c in stage.conditions} == {
            c.id for c in original_problem.conditions
        }
        # Measurements are filtered to the stage's time horizon
        assert all(m.time <= t_end for m in stage.measurements)
        assert any(m.time == t_end for m in stage.measurements)


# -----------------------------------------------------------------------------
# Problem 3: Experiments with partial time coverage
# -----------------------------------------------------------------------------
def test_partial_experiments(dir_tmp: Path) -> None:
    """Test that experiments and conditions are correctly filtered per stage.

    exp2/condB only have late measurements, so they should be absent in early
    stages. condC belongs to exp1's period at t=4, so it is also absent in
    stages where t_end < 4.0.
    """
    path_yaml = dir_tmp / "problem.yaml"
    get_prob_partial_experiments(dir_tmp)
    dir_uniform = dir_tmp / "dir_uniform"
    training_problem = PEtabTrainingProblem(
        yaml=path_yaml, strategy=CurriculumLearning(UniformPartition(n=3))
    )
    training_problem.export(dir_uniform)
    end_times = [1.0, 3.0, 6.0]

    for i, t_end in enumerate(end_times):
        stage = petab.v2.Problem.from_yaml(
            dir_uniform / f"stage{i + 1}" / "problem.yaml"
        )
        # Measurements are filtered to the stage's time horizon
        assert all(m.time <= t_end for m in stage.measurements)
        assert any(m.time == t_end for m in stage.measurements)

        if t_end < 4.0:
            # Only exp1 survives, and only its t=0 period is within range
            assert {e.id for e in stage.experiments} == {"exp1"}
            assert {c.id for c in stage.conditions} == {"condA"}
        else:
            # Both experiments survive, all periods within range
            assert {e.id for e in stage.experiments} == {"exp1", "exp2"}
            assert {c.id for c in stage.conditions} == {"condA", "condB", "condC"}
