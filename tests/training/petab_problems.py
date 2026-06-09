from pathlib import Path

import petab.v2
from petab.v1.models.sbml_model import SbmlModel
from petab.v2.core import ProblemConfig


def get_prob_no_experiment(dir_tmp: Path) -> None:
    """Create and export a PEtab problem without experiment table."""
    sbml_path = Path(__file__).parent / "model.xml"
    model = SbmlModel.from_file(sbml_path, model_id="test_model")

    prob_no_experiments = petab.v2.Problem()
    prob_no_experiments.add_observable("obs1", formula="x1")
    prob_no_experiments.add_observable("obs2", formula="x2")
    prob_no_experiments.add_parameter("k1", lb=1e-3, ub=1e3)
    prob_no_experiments.add_parameter("k2", lb=1e-3, ub=1e3)
    prob_no_experiments.models = [model]
    for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        prob_no_experiments.add_measurement("obs1", time=t, measurement=t * 0.5)
        prob_no_experiments.add_measurement("obs2", time=t, measurement=t * 0.3)

    prob_no_experiments.measurement_tables[0].rel_path = Path("measurements.tsv")
    prob_no_experiments.observable_tables[0].rel_path = Path("observables.tsv")
    prob_no_experiments.parameter_tables[0].rel_path = Path("parameters.tsv")
    prob_no_experiments.models[0].rel_path = Path("model.xml")

    prob_no_experiments.config = ProblemConfig(
        format_version="2.0.0", base_path=dir_tmp, filepath=dir_tmp / "problem.yaml"
    )
    prob_no_experiments.to_files(dir_tmp)
    return None


def get_prob_all_experiments(dir_tmp: Path) -> None:
    """Create a PEtab problem where all experiments cover all time points."""
    sbml_path = Path(__file__).parent / "model.xml"
    model = SbmlModel.from_file(sbml_path, model_id="test_model")

    prob = petab.v2.Problem()
    prob.add_observable("obs1", formula="x1")
    prob.add_parameter("k1", lb=1e-3, ub=1e3)
    prob.add_condition("condA", x1=1.0)
    prob.add_condition("condB", x1=2.0)
    prob.add_experiment("exp1", 0.0, "condA")
    prob.add_experiment("exp2", 0.0, "condB")
    prob.models = [model]
    for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        prob.add_measurement("obs1", time=t, measurement=t * 0.5, experiment_id="exp1")
        prob.add_measurement("obs1", time=t, measurement=t * 0.6, experiment_id="exp2")

    prob.measurement_tables[0].rel_path = Path("measurements.tsv")
    prob.observable_tables[0].rel_path = Path("observables.tsv")
    prob.parameter_tables[0].rel_path = Path("parameters.tsv")
    prob.condition_tables[0].rel_path = Path("conditions.tsv")
    prob.experiment_tables[0].rel_path = Path("experiments.tsv")
    prob.config = ProblemConfig(
        format_version="2.0.0",
        base_path=dir_tmp,
        filepath=dir_tmp / "problem.yaml",
    )
    prob.to_files(dir_tmp)
    return None


def get_prob_partial_experiments(dir_tmp: Path) -> None:
    """Create a PEtab problem where experiments have partial time coverage."""
    sbml_path = Path(__file__).parent / "model.xml"
    model = SbmlModel.from_file(sbml_path, model_id="test_model")

    prob = petab.v2.Problem()
    prob.add_observable("obs1", formula="x1")
    prob.add_parameter("k1", lb=1e-3, ub=1e3)
    prob.add_condition("condA", x1=1.0)
    prob.add_condition("condB", x1=2.0)
    prob.add_condition("condC", x1=3.0)
    prob.add_experiment("exp1", 0.0, "condA", 4.0, "condC")
    prob.add_experiment("exp2", 0.0, "condB")
    prob.models = [model]
    for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        prob.add_measurement("obs1", time=t, measurement=t * 0.5, experiment_id="exp1")
    for t in [4.0, 5.0, 6.0]:
        prob.add_measurement("obs1", time=t, measurement=t * 0.6, experiment_id="exp2")

    prob.measurement_tables[0].rel_path = Path("measurements.tsv")
    prob.observable_tables[0].rel_path = Path("observables.tsv")
    prob.parameter_tables[0].rel_path = Path("parameters.tsv")
    prob.condition_tables[0].rel_path = Path("conditions.tsv")
    prob.experiment_tables[0].rel_path = Path("experiments.tsv")
    prob.config = ProblemConfig(
        format_version="2.0.0",
        base_path=dir_tmp,
        filepath=dir_tmp / "problem.yaml",
    )
    prob.to_files(dir_tmp)


def get_prob_partial_experiments_ms(dir_tmp: Path) -> None:
    """Create a PEtab problem where one of two experiments lacks measurements in the last window.

    Used for testing multiple shooting.
    """
    sbml_path = Path(__file__).parent / "model.xml"
    model = SbmlModel.from_file(sbml_path, model_id="test_model")
    prob = petab.v2.Problem()
    prob.add_observable("obs1", formula="x1")
    prob.add_parameter("k1", lb=1e-3, ub=1e3)
    prob.add_condition("condA", x1=1.0)
    prob.add_condition("condB", x1=2.0)
    prob.add_experiment("exp1", 0.0, "condA")
    prob.add_experiment("exp2", 0.0, "condB")
    prob.models = [model]
    for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        prob.add_measurement("obs1", time=t, measurement=t * 0.5, experiment_id="exp1")
    for t in [0.0, 1.0, 2.0]:
        prob.add_measurement("obs1", time=t, measurement=t * 0.6, experiment_id="exp2")
    prob.measurement_tables[0].rel_path = Path("measurements.tsv")
    prob.observable_tables[0].rel_path = Path("observables.tsv")
    prob.parameter_tables[0].rel_path = Path("parameters.tsv")
    prob.condition_tables[0].rel_path = Path("conditions.tsv")
    prob.experiment_tables[0].rel_path = Path("experiments.tsv")
    prob.config = ProblemConfig(
        format_version="2.0.0",
        base_path=dir_tmp,
        filepath=dir_tmp / "problem.yaml",
    )
    prob.to_files(dir_tmp)
