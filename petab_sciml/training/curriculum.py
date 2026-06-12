"""Curriculum learning export for PEtab problems."""

import copy
from dataclasses import dataclass
from pathlib import Path

import petab

from .partition import Partition
from ._helper import _resolve_output_dir


@dataclass
class CurriculumLearningProblem:
    """Curriculum learning training problem.

    Training difficulty is progressively increased across stages by gradually
    extending the measurement time horizon. Each stage is a self-contained
    PEtab problem containing all measurements up to the stage's end time point.

    Parameters
    ----------
    yaml:
        Path to the source PEtab YAML file.
    partition:
        How to split the time range in the PEtab measurement table into
        curriculum stages. End points are provided by ``UniformPartition(n)``
        for uniform stages or ``CustomPartition`` for finer control.

    Example
    -------
    For measurements at time points ``[0, 1, 2, 3, 4, 5, 6]``, a uniform
    partition of 3 stages produces stages with time horizons ending at
    ``[1, 3, 6]``:

    >>> problem = CurriculumLearningProblem(
    ...     yaml="my_model/problem.yaml",
    ...     partition=UniformPartition(n=3),
    ... )
    >>> problem.export()
    """

    yaml: Path | str
    partition: Partition

    def export(
        self, output_dir: Path | str | None = None, validate: bool = True
    ) -> None:
        """Export this curriculum learning problem to disk.

        Creates one sub-directory per curriculum stage under ``output_dir``,
        each containing a self-contained PEtab problem with measurements
        filtered to the stage's time horizon.

        Parameters
        ----------
        output_dir:
            Directory to write the exported stage problems to. Created if it
            does not exist. If ``None``, defaults to a ``cl_prob`` subdirectory
            of the source YAML's directory.
        validate:
            Whether to validate each exported stage problem before writing.
        """
        output_dir = _resolve_output_dir(
            yaml=self.yaml, output_dir=output_dir, default_name="cl_prob"
        )

        problem = petab.v2.Problem.from_yaml(self.yaml)
        measurement_df = problem.measurement_df
        stage_end_times = self.partition.get_time_points(measurement_df)

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
                            c
                            for c in table.conditions
                            if c.id in surviving_condition_ids
                        ],
                        rel_path=table.rel_path,
                    )
                    for table in stage_problem.condition_tables
                ]

            stage_problem.to_files(base_path=stage_dir)

            if validate:
                stage_problem.validate()
