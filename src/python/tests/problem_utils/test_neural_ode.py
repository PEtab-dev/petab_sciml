from contextlib import chdir
import csv
from petab_sciml.problem_utils.neural_ode import (
    generate_neural_ode_problem, 
    write_remaining_petab_files
)
import os
from yaml import safe_load


def test_generate_neural_ode_problem(tmp_path):
    """generate_neural_ode_problem"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        species = ["prey", "predator"]
        generate_neural_ode_problem(species)

        expected_files = [
            "model.xml",
            "hybridization.tsv",
            "mapping.tsv",
            "parameters.tsv",
        ]
        for fname in expected_files:
            assert os.path.isfile(fname)

        hybridization_expected = [
            {"targetId": "net1_input0", "targetValue": "prey"},
            {"targetId": "net1_input1", "targetValue": "predator"},
            {"targetId": "prey_param", "targetValue": "net1_output0"},
            {"targetId": "predator_param", "targetValue": "net1_output1"},
        ]
        with open("hybridization.tsv", "r") as f:
            hybdridization = list(csv.DictReader(f, delimiter="\t"))
            assert hybdridization == hybridization_expected

        mapping_expected = [
            {"petabEntityId": "net1_input0", "modelEntityId": "net1.inputs[0][0]"},
            {"petabEntityId": "net1_input1", "modelEntityId": "net1.inputs[0][1]"},
            {"petabEntityId": "net1_output0", "modelEntityId": "net1.outputs[0][0]"},
            {"petabEntityId": "net1_output1", "modelEntityId": "net1.outputs[0][1]"},
            {"petabEntityId": "net1_ps", "modelEntityId": "net1.parameters"},
        ]
        with open("mapping.tsv", "r") as f:
            mapping = list(csv.DictReader(f, delimiter="\t"))
            assert mapping == mapping_expected

        parameters_expected = [
            {
                "parameterId": "net1_ps",
                "parameterScale": "lin",
                "lowerBound": "-inf",
                "upperBound": "inf",
                "nominalValue": "",
                "estimate": "1",
            }
        ]
        with open("parameters.tsv", "r") as f:
            parameters = list(csv.DictReader(f, delimiter="\t"))
            assert parameters == parameters_expected

def test_generate_neural_ode_problem_with_options(tmp_path):
    """generate_neural_ode_problem"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        species = ["prey", "predator"]
        generate_neural_ode_problem(species, model_filename="lv.xml", network_name="mynet")

        expected_files = [
            "lv.xml",
            "hybridization.tsv",
            "mapping.tsv",
            "parameters.tsv",
        ]
        for fname in expected_files:
            assert os.path.isfile(fname)

        mapping_expected = [
            {"petabEntityId": "mynet_input0", "modelEntityId": "mynet.inputs[0][0]"},
            {"petabEntityId": "mynet_input1", "modelEntityId": "mynet.inputs[0][1]"},
            {"petabEntityId": "mynet_output0", "modelEntityId": "mynet.outputs[0][0]"},
            {"petabEntityId": "mynet_output1", "modelEntityId": "mynet.outputs[0][1]"},
            {"petabEntityId": "mynet_ps", "modelEntityId": "mynet.parameters"},
        ]
        with open("mapping.tsv", "r") as f:
            mapping = list(csv.DictReader(f, delimiter="\t"))
            assert mapping == mapping_expected


def test_write_remaining_petab_files(tmp_path):
    """write_remaining_petab_files"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        species = ["prey", "predator"]
        generate_neural_ode_problem(species)

        _write_test_measurements_file()
        measurements_filename = "measurements.tsv"
        observables_filename = "observables.tsv"
        network_filename = "net1.yaml"
        array_filenames = ["supernet_params.h5", "supernet_inputs.h5"]

        write_remaining_petab_files(
            "model.xml",
            measurements_filename,
            observables_filename,
            network_filename,
            array_filenames
        )

        expected_files = [
            "conditions.tsv",
            "problem.yaml",
        ]
        for fname in expected_files:
            assert os.path.isfile(fname)

        conditions_expected = [{"conditionId": "cond1"}]
        with open("conditions.tsv", "r") as f:
            condition = list(csv.DictReader(f, delimiter="\t"))
            assert condition == conditions_expected

        with open("problem.yaml", "r") as f:
            problem = safe_load(f)
            assert (
                problem["problems"][0]["model_files"]["model"]["location"]
                == "model.xml"
            )
            assert (
                problem["problems"][0]["measurement_files"][0]
                == "measurements.tsv"
            )
            assert (
                "net1" in problem["extensions"]["sciml"]["neural_nets"].keys()
            )
            assert (
                problem["extensions"]["sciml"]["neural_nets"]["net1"]["location"]
                == "net1.yaml"
            )
            assert problem["extensions"]["sciml"]["array_files"] == array_filenames

def _write_test_measurements_file():
    """Write a simple measurements file for testing purposes."""
    measurements = [
        {"observableId": "prey_o", "simulationConditionId": "cond1", "time": "1.0", "measurement": "0.1"},
        {"observableId": "prey_o", "simulationConditionId": "cond1", "time": "2.0", "measurement": "0.5"},
        {"observableId": "predator_o", "simulationConditionId": "cond1", "time": "1.0", "measurement": "0.8"},
        {"observableId": "predator_o", "simulationConditionId": "cond1", "time": "1.0", "measurement": "0.2"},
    ]
    with open("measurements.tsv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=measurements[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(measurements)