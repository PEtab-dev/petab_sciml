"""Partition the time-axis for training strategies"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UniformPartition:
    """Partition a time range into equally-spaced segments.

    Interior split points are computed by dividing the unique time points in
    the PEtab measurement table into ``n`` equally-sized groups. Note that
    this partitions time points, not individual measurements, so the number
    of measurements per segment may vary.

    Used by :class:`CurriculumLearningProblem`,
    :class:`MultipleShootingProblem`, and
    :class:`CurriculumMultipleShootingProblem`; see each class for what the
    resulting segments mean.

    Parameters
    ----------
    n:
        Number of segments. Must be >= 2.
    """

    n: int

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"n must be >= 2, got {self.n}.")

    def get_time_points(self, measurement_table: pd.DataFrame) -> list[float]:
        """Return the end time of each segment for this partition.

        Parameters
        ----------
        measurement_table:
            PEtab measurement table providing the time points to partition.

        Returns
        -------
        The end time of each segment, including the final time point of the data.
        """
        time_points = np.sort(measurement_table["time"].unique())

        if len(time_points) < self.n:
            raise ValueError(
                f"Cannot create {self.n} segments with only "
                f"{len(time_points)} unique time points."
            )

        # Divide time points into n groups as evenly as possible, with any
        # remainder distributed across the last few groups
        n = self.n
        base_size = len(time_points) // n
        remainder = len(time_points) % n
        sizes = [base_size + (1 if i >= n - remainder else 0) for i in range(n)]
        indices = np.cumsum(sizes)
        return [float(time_points[i - 1]) for i in indices]


@dataclass
class CustomPartition:
    """Partition a time range at explicit interior split points.

    The start and end of the time range are inferred from the PEtab measurement
    table at export time, so only interior points should be provided.

    The interpretation of the split points depends on the training strategy;
    see :class:`CurriculumLearningProblem`, :class:`MultipleShootingProblem`,
    and :class:`CurriculumMultipleShootingProblem` for details.

    Parameters
    ----------
    interior_points:
        Strictly increasing list of interior split points. Must not include
        the start or end of the time range.

    Examples
    --------
    ``CustomPartition([3.0, 4.0, 5.0])`` on data spanning ``[0, 6]`` produces
    four segments with end points ``[3, 4, 5, 6]``.
    """

    interior_points: list[float]

    def __post_init__(self):
        if not all(
            self.interior_points[i] < self.interior_points[i + 1]
            for i in range(len(self.interior_points) - 1)
        ):
            raise ValueError(
                "interior_points must be strictly increasing, got "
                f"{self.interior_points}."
            )

    def get_time_points(self, measurement_table: pd.DataFrame) -> list[float]:
        """Return the end time of each segment for this partition.

        Parameters
        ----------
        measurement_table:
            PEtab measurement table providing the time points to partition.

        Returns
        -------
        The interior points followed by the final time point of the data.
        """
        time_points = np.sort(measurement_table["time"].unique())
        t_end = float(time_points[-1])

        if any(p >= t_end for p in self.interior_points):
            raise ValueError(
                "interior_points must be strictly less than the maximum time "
                f"point {t_end}, got {self.interior_points}."
            )

        all_points = [float(time_points[0])] + self.interior_points + [t_end]
        for i in range(len(all_points) - 1):
            t_start_segment, t_end_segment = all_points[i], all_points[i + 1]
            if not any(t_start_segment <= t <= t_end_segment for t in time_points):
                raise ValueError(
                    f"Segment [{t_start_segment}, {t_end_segment}] contains no "
                    "data points. The corresponding curriculum learning stage or "
                    "multiple shooting window would therefore not introduce any "
                    "new measurements."
                )

        return self.interior_points + [t_end]


Partition = UniformPartition | CustomPartition
