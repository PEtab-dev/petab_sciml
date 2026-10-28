"""Curriculum multiple shooting export for PEtab problems."""

import copy
from dataclasses import dataclass
from pathlib import Path

import petab.v2

from ._helper import _resolve_output_dir
from .multiple_shooting import _get_ms_problem
from .partition import Partition


@dataclass
class CurriculumMultipleShootingProblem:
    """Curriculum multiple shooting training problem.

    Training starts with a full multiple shooting configuration and
    progressively merges adjacent windows across stages until a single window
    remains. Each stage is a self-contained multiple shooting problem with one
    fewer window than the previous stage. At each stage, a continuity penalty
    as in multiple shooting is applied at the first overlapping time point
    between adjacent windows.

    Parameters
    ----------
    yaml, partition, penalty, log_penalty, initial_value:
        See :class:`MultipleShootingProblem` for details; parameters have the
        same interpretation and apply across all curriculum stages.

    Examples
    --------
    With ``UniformPartition(3)`` on data spanning ``[0, 6]``, the initial
    windows are ``[0, 2], [2, 4], [4, 6]``. The stages are then:

    - Stage 1: windows ``[0, 2], [2, 4], [4, 6]`` (full multiple shooting)
    - Stage 2: windows ``[0, 4], [2, 6]`` (last window dropped, remaining extended)
    - Stage 3: window ``[0, 6]`` (single window, original problem)

    >>> problem = CurriculumMultipleShootingProblem(
    ...     yaml="my_model/problem.yaml",
    ...     partition=UniformPartition(n=3),
    ...     penalty=10.0,
    ... )
    >>> problem.export()
    """

    yaml: Path | str
    partition: Partition
    penalty: float
    log_penalty: bool = False
    initial_value: float = 1e-6

    def export(
        self, output_dir: Path | str | None = None, validate: bool = True
    ) -> None:
        """Export this curriculum multiple shooting problem to disk.

        Creates one sub-directory per curriculum stage under ``output_dir``,
        each containing a self-contained PEtab problem. Each stage is itself a
        PEtab problem.

        Parameters
        ----------
        output_dir:
            Directory to write the exported stage problems to. Created if it
            does not exist. If ``None``, defaults to a ``cms_prob``
            subdirectory of the source YAML's directory.
        validate:
            Whether to validate each exported stage problem before writing.
        """
        output_dir = _resolve_output_dir(
            yaml=self.yaml, output_dir=output_dir, default_name="ms_prob"
        )

        problem = petab.v2.Problem.from_yaml(self.yaml)

        measurement_df = problem.measurement_df
        end_times = self.partition.get_time_points(measurement_df)
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
                stage_problem = _get_ms_problem(problem, stage_windows, self)

            if validate:
                stage_problem.validate()

            stage_dir = output_dir / f"stage{stage_idx + 1}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_problem.to_files(stage_dir)
