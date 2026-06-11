"""Multiple shooting export for PEtab problems."""

import copy
import math
from pathlib import Path

import petab.v2
from petab.v1.models.sbml_model import SbmlModel

from .partition import get_partition_time_points


def _export_multiple_shooting(
    problem: petab.v2.Problem, strategy, output_dir: Path, validate: bool
) -> None:
    """Export a PEtab problem as a multiple shooting training problem.

    Creates a transformed PEtab problem under ``output_dir`` where the time
    span is split into shooting windows. Each window-experiment pair has its
    own estimated initial state, and a quadratic continuity penalty
    encourages a continuous trajectory between adjacent windows.

    Boundary measurements are double-counted: a measurement at exactly the
    boundary between two windows appears in both windows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_df = problem.measurement_df
    end_times = get_partition_time_points(strategy.partition, measurement_df)
    t_min = float(measurement_df["time"].min())
    windows = list(zip([t_min] + end_times[:-1], end_times))

    ms_problem = _get_ms_problem(problem, windows, strategy)
    if validate:
        ms_problem.validate()

    ms_problem.to_files(output_dir)


def _get_ms_problem(
    problem: petab.v2.Problem,
    windows: list[float],
    strategy,
) -> petab.v2.Problem:
    """Build multiple shooting PEtab problem given windows

    Used to build both multiple shooting and curriculum multiple shooting
    problems.
    """

    _check_no_preequilibration(problem)
    if not isinstance(problem.model, SbmlModel):
        raise ValueError("Multiple shooting export currently requires an SBML model.")
    specie_ids = [s.getId() for s in problem.model.sbml_model.getListOfSpecies()]

    t_min = windows[0][0]
    n_windows = len(windows)

    ms_problem = copy.deepcopy(problem)
    _ensure_default_experiment(ms_problem, t_min)

    original_experiments = list(ms_problem.experiments)
    original_exp_t_end = {
        exp.id: max(
            (m.time for m in ms_problem.measurements if m.experiment_id == exp.id),
            default=float("-inf"),
        )
        for exp in original_experiments
    }

    # MS_PENALTY_SQRT is the only globally-shared addition
    ms_problem.parameter_tables[0].elements.append(
        petab.v2.Parameter(
            id="MS_PENALTY_SQRT",
            nominal_value=math.sqrt(strategy.penalty),
            estimate=False,
        )
    )

    if n_windows > 1 and not ms_problem.condition_tables[0].rel_path:
        ms_problem.condition_tables[0].rel_path = Path("conditions.tsv")

    new_measurements = []
    new_experiments = []
    for window_index, (t0, tf) in enumerate(windows):
        for orig_exp in original_experiments:
            if original_exp_t_end[orig_exp.id] < t0:
                continue

            new_exp_id = f"WINDOW{window_index}_EXPERIMENT_{orig_exp.id}"

            # For windows i > 0, this experiment needs its own ICs per window
            if window_index > 0:
                _add_per_experiment_ic_artifacts(
                    ms_problem, orig_exp.id, window_index, specie_ids, strategy
                )

            new_measurements.extend(
                _build_window_measurements(ms_problem, orig_exp, new_exp_id, t0, tf)
            )
            new_experiments.append(
                _build_window_experiment(orig_exp, new_exp_id, window_index, t0, tf)
            )

            # Add penalty measurements only if this experiment continues into
            # the next window
            if window_index < n_windows - 1:
                next_t0 = windows[window_index + 1][0]
                if original_exp_t_end[orig_exp.id] >= next_t0:
                    new_measurements.extend(
                        _build_penalty_measurements(
                            new_exp_id,
                            orig_exp.id,
                            window_index + 1,
                            next_t0,
                            specie_ids,
                        )
                    )

    ms_problem.measurement_tables[0].elements = new_measurements
    ms_problem.experiment_tables[0].elements = new_experiments
    return ms_problem


def _check_no_preequilibration(problem: petab.v2.Problem) -> None:
    """Raise if any experiment has a pre-equilibration period."""
    for exp in problem.experiments:
        if exp.has_preequilibration:
            raise ValueError(
                f"Multiple shooting does not support pre-equilibration "
                f"periods, found in experiment '{exp.id}'."
            )


def _ensure_default_experiment(problem: petab.v2.Problem, t_min: float) -> None:
    """If the problem has no experiments, add a default one and assign it.

    All measurements without an ``experiment_id`` are reassigned to the new
    default experiment.
    """
    if problem.experiments:
        return None
    default_exp_id = "default_experiment"
    problem.add_experiment(default_exp_id, t_min, [])
    for table in problem.measurement_tables:
        for m in table.measurements:
            if m.experiment_id is None:
                m.experiment_id = default_exp_id

    if not problem.experiment_tables[0].rel_path:
        problem.experiment_tables[0].rel_path = Path("experiments.tsv")
    return None


def _add_per_experiment_ic_artifacts(
    problem: petab.v2.Problem,
    orig_exp_id: str,
    window_index: int,
    specie_ids: list[str],
    strategy,
) -> None:
    """Add per-(window, experiment) IC parameters, penalty observables, and IC condition.

    For each species, adds an estimated parameter holding the initial value
    of that species at the start of the window for this experiment, a
    penalty observable encoding the squared deviation between the simulated
    species and the estimated initial value, and a condition that sets the
    species to the estimated initial value at the start of the window.
    """
    prefix = f"WINDOW{window_index}_EXPERIMENT_{orig_exp_id}"
    for specie_id in specie_ids:
        problem.parameter_tables[0].elements.append(
            petab.v2.Parameter(
                id=f"{prefix}_PARAMETER_{specie_id}",
                lb=1e-8,
                ub=1e8,
                nominal_value=strategy.initial_value,
                estimate=True,
            )
        )

        if strategy.log_penalty:
            _formula = f"(log(abs({specie_id})) - log(abs({prefix}_PARAMETER_{specie_id}))) * MS_PENALTY_SQRT"
        else:
            _formula = (
                f"({specie_id} - {prefix}_PARAMETER_{specie_id}) * MS_PENALTY_SQRT"
            )
        problem.observable_tables[0].elements.append(
            petab.v2.Observable(
                id=f"{prefix}_PENALTY_{specie_id}",
                formula=_formula,
                noise_formula="1.0",
                noise_distribution="normal",
            )
        )
    problem.condition_tables[0].elements.append(
        petab.v2.Condition(
            id=f"{prefix}_IC",
            changes=[
                petab.v2.Change(
                    target_id=specie_id,
                    target_value=f"{prefix}_PARAMETER_{specie_id}",
                )
                for specie_id in specie_ids
            ],
        )
    )


def _build_window_measurements(
    problem: petab.v2.Problem,
    orig_exp: petab.v2.Experiment,
    new_exp_id: str,
    t0: float,
    tf: float,
) -> list[petab.v2.Measurement]:
    """Copy original measurements within [t0, tf] under the new experiment ID.

    Boundary measurements (at exactly t0 or tf) are included and may
    therefore appear in adjacent windows. This in order to make each window
    aware, and leverage all relevant data-points
    """
    return [
        petab.v2.Measurement(
            observable_id=m.observable_id,
            experiment_id=new_exp_id,
            time=m.time,
            measurement=m.measurement,
            observable_parameters=list(m.observable_parameters),
            noise_parameters=list(m.noise_parameters),
        )
        for m in problem.measurements
        if m.experiment_id == orig_exp.id and t0 <= m.time <= tf
    ]


def _build_window_experiment(
    orig_exp: petab.v2.Experiment,
    new_exp_id: str,
    window_index: int,
    t0: float,
    tf: float,
) -> petab.v2.Experiment:
    """Build the new experiment for a (window, original experiment) pair.

    First window keeps original periods within [t0, tf]; an empty period at
    t0 is added if none exists, so the simulation starts at the window's
    start time. Subsequent windows start with the window's IC condition at
    t0 and include original periods within (t0, tf].
    """
    if window_index == 0:
        leading_period = (
            None
            if any(p.time == t0 for p in orig_exp.periods)
            else petab.v2.ExperimentPeriod(time=t0, condition_ids=[])
        )
    else:
        leading_period = petab.v2.ExperimentPeriod(
            time=t0,
            condition_ids=[f"WINDOW{window_index}_EXPERIMENT_{orig_exp.id}_IC"],
        )

    include_t0 = window_index == 0
    periods = [leading_period] if leading_period is not None else []
    periods.extend(
        petab.v2.ExperimentPeriod(time=p.time, condition_ids=list(p.condition_ids))
        for p in orig_exp.periods
        if (t0 <= p.time if include_t0 else t0 < p.time) and p.time <= tf
    )
    return petab.v2.Experiment(id=new_exp_id, periods=periods)


def _build_penalty_measurements(
    new_exp_id: str,
    orig_exp_id: str,
    next_window_index: int,
    next_t0: float,
    specie_ids: list[str],
) -> list[petab.v2.Measurement]:
    """Build penalty measurements at next_t0 pointing to next window's per-experiment parameters.

    Each measurement encodes the continuity penalty for one species at the
    boundary between this window and the next one. The measurement value is
    0 since we want the penalty to vanish at the optimum.
    """
    next_prefix = f"WINDOW{next_window_index}_EXPERIMENT_{orig_exp_id}"
    return [
        petab.v2.Measurement(
            observable_id=f"{next_prefix}_PENALTY_{specie_id}",
            experiment_id=new_exp_id,
            time=next_t0,
            measurement=0.0,
        )
        for specie_id in specie_ids
    ]
