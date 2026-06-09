from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import petab

from .partition import Partition
from .curriculum import _export_curriculum_learning


@dataclass
class CurriculumLearning:
    """Curriculum learning training strategy.

    Training difficulty is progressively increased across stages by gradually
    extending the measurement time horizon. Each stage is a self-contained
    PEtab problem, containing all measurements up to the stage's end time point.

    For training loops, the parameter estimate from stage ``i`` should be used
    to initialise stage ``i+1``.

    Parameters
    ----------
    partition:
        How to split the time range in the PEtab measurement table into
        curriculum stages. A ``UniformPartition(n)`` produces ``n`` stages by
        dividing the unique time points in the measurement table into
        equally-sized groups, where the end of each group defines the stage's
        time horizon. A ``CustomPartition`` allows finer control by specifying
        the end time point of each stage explicitly.
    """

    partition: Partition


@dataclass
class MultipleShooting:
    """Multiple shooting training strategy.

    The simulation time span is split into windows that are fitted jointly.
    Each window has its own estimated initial state, and a continuity penalty
    encourages a continuous trajectory between adjacent windows.

    Parameters
    ----------
    partition:
        How to split the time range into shooting windows. A
        ``UniformPartition(n)`` produces ``n`` windows by dividing the unique
        time points in the measurement table into equally-sized groups. A
        ``CustomPartition`` allows finer control by specifying the end-points
        of all windows except the last.
    penalty:
        Weight ``lambda`` of the continuity penalty term. A quadratic penalty
        is applied, so the contribution to the loss is
        ``lambda * (state - estimated_initial_state) ** 2``.
    log_penalty:
        If ``True``, apply the penalty on log-transformed states;
        ``lambda * (log(state) - log(estimated_initial_state)) ** 2``. Useful
        when states span several orders of magnitude.
    initial_value:
        Initial guess for the estimated initial state of each window, applied
        uniformly across all model states and all windows except the first
        (whose initial states are already defined in the PEtab problem).
    """

    partition: Partition
    penalty: float
    log_penalty: bool = False
    initial_value: float = 1e-6


@dataclass
class CurriculumMultipleShooting:
    """Curriculum multiple shooting training strategy.

    Training starts with a full multiple shooting configuration and
    progressively merges adjacent windows across stages until a single window
    remains. Each stage is a self-contained multiple shooting problem with one
    fewer window than the previous stage.

    For example, with ``UniformPartition(3)`` on data spanning ``[0, 6]``, the
    initial windows are ``[0, 2], [2, 4], [4, 6]``. The stages are then:

    - Stage 1: windows ``[0, 2], [2, 4], [4, 6]`` (full multiple shooting)
    - Stage 2: windows ``[0, 4], [2, 6]`` (last window dropped, remaining extended)
    - Stage 3: window ``[0, 6]`` (single window, original problem)

    At each stage, a continuity penalty as in multiple shooting is applied at
    the first overlapping time point between adjacent windows.

    Parameters
    ----------
    multiple_shooting:
        The initial multiple shooting configuration, defining the starting
        number of windows and the penalty applied at each stage. The number
        of curriculum stages is determined by the number of windows, with
        the final stage corresponding to the original PEtab problem.
    """

    multiple_shooting: MultipleShooting


Strategy = MultipleShooting | CurriculumLearning | CurriculumMultipleShooting


@dataclass
class PEtabTrainingProblem:
    """A PEtab problem wrapped with a training strategy, ready for export.

    Constructs a transformed PEtab problem (or sequence of problems for
    curriculum-based strategies) from an existing PEtab YAML and a training
    strategy specification. The source problem is never modified; all output
    is written to a new directory on export.

    Parameters
    ----------
    yaml:
        Path to the source PEtab YAML file.
    strategy:
        Training strategy to apply. One of :class:`MultipleShooting`,
        :class:`CurriculumLearning`, or :class:`CurriculumMultipleShooting`.

    Examples
    --------
    Multiple shooting with uniform windows::

        problem = PEtabTrainingProblem(
            yaml="my_model/problem.yaml",
            strategy=MultipleShooting(partition=UniformPartition(5), penalty=10.0),
        )
        problem.export("my_model_ms/", validate = True)

    Curriculum learning with custom stage boundaries::

        problem = PEtabTrainingProblem(
            yaml="my_model/problem.yaml",
            strategy=CurriculumLearning(partition=CustomPartition([2.0, 4.0])),
        )
        problem.export("my_model_cl/", validate = True)
    """

    yaml: Path | str
    strategy: Strategy

    def export(self, output_dir: Path | str = None, validate: bool = True) -> None:
        """Transform and export the PEtab problem to ``output_dir``.

        Applies the training strategy to the source PEtab problem and writes
        the resulting problem(s) to ``output_dir``. Calls :meth:`validate`
        before writing anything.

        For :class:`MultipleShooting`, a single transformed PEtab problem is
        written. For :class:`CurriculumLearning` and
        :class:`CurriculumMultipleShooting`, one sub-directory per stage is
        written under ``output_dir``.

        Parameters
        ----------
        output_dir:
            Directory to write the exported problem(s) to. Created if it does
            not exist. Must be different from the source problem's directory.
        validate:
            Whether to validate the PEtab problem(s) before exporting.
        """
        problem = petab.v2.Problem.from_yaml(self.yaml)
        if output_dir is None:
            output_dir = Path(self.yaml).parent
        output_dir = Path(output_dir)

        if isinstance(self.strategy, CurriculumLearning):
            _export_curriculum_learning(problem, self.strategy, output_dir, validate)
