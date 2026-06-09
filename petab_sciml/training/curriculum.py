"""Curriculum learning export for PEtab problems."""

import copy
from pathlib import Path

import petab

from .partition import get_partition_time_points


def _export_curriculum_learning(
    problem: petab.v2.Problem, strategy, output_dir: Path, validate: bool
) -> None:
    """Export a PEtab problem as a curriculum learning training problem.

    Creates one sub-directory per curriculum stage under ``output_dir``,
    each containing a self-contained PEtab problem with measurements filtered
    to the stage's time horizon. Experiments and conditions not referenced
    by the filtered measurement table are removed from each stage problem.
    """
    measurement_df = problem.measurement_df
    stage_end_times = get_partition_time_points(strategy.partition, measurement_df)

    for i, t_end in enumerate(stage_end_times):
        stage_dir = output_dir / f"stage{i + 1}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_problem = copy.deepcopy(problem)

        stage_problem.measurement_tables = [
            petab.v2.MeasurementTable(
                elements=[m for m in table.measurements if m.time <= t_end],
                rel_path=table.rel_path,
            )
            for table in stage_problem.measurement_tables
        ]

        # Identify surviving condition and experiment IDs (if relevant)
        if problem.experiment_df is not None:
            surviving_experiment_ids = {
                m.experiment_id
                for table in stage_problem.measurement_tables
                for m in table.measurements
            }
            stage_problem.experiment_tables = [
                petab.v2.ExperimentTable(
                    elements=[
                        petab.v2.Experiment(
                            id=e.id,
                            periods=[p for p in e.periods if p.time <= t_end],
                        )
                        for e in table.experiments
                        if e.id in surviving_experiment_ids
                    ],
                    rel_path=table.rel_path,
                )
                for table in stage_problem.experiment_tables
            ]

            surviving_condition_ids = {
                cid
                for table in stage_problem.experiment_tables
                for experiment in table.experiments
                for period in experiment.periods
                if period.time <= t_end
                for cid in period.condition_ids
            }
            stage_problem.condition_tables = [
                petab.v2.ConditionTable(
                    elements=[
                        c for c in table.conditions if c.id in surviving_condition_ids
                    ],
                    rel_path=table.rel_path,
                )
                for table in stage_problem.condition_tables
            ]

        stage_problem.to_files(base_path=stage_dir)

        if validate:
            stage_problem.validate()

    return None
