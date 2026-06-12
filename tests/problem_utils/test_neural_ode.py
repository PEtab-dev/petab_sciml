import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from libsbml import readSBML
from torch import nn
from yaml import safe_load

from petab_sciml.problem_utils.neural_ode import (
    create_neural_ode,
    create_neural_ode_problem,
)
from petab_sciml.standard.nn_model import Input, NNModel, NNModelStandard


def test_create_neural_ode(dir_tmp: Path):
    """create_neural_ode"""

    test_dir = dir_tmp / "petab"
    test_dir.mkdir()

    species = ["prey", "predator"]
    create_neural_ode(species, save_directory=test_dir)

    expected_files = [
        "model.xml",
        "hybridization.tsv",
        "mapping.tsv",
        "parameters.tsv",
    ]
    for fname in expected_files:
        assert (test_dir / fname).is_file()

    hybridization_expected = [
        {"targetId": "net1_input0", "targetValue": "prey"},
        {"targetId": "net1_input1", "targetValue": "predator"},
        {"targetId": "prey_param", "targetValue": "net1_output0"},
        {"targetId": "predator_param", "targetValue": "net1_output1"},
    ]
    with open(test_dir / "hybridization.tsv", "r") as f:
        hybdridization = list(csv.DictReader(f, delimiter="\t"))
        assert hybdridization == hybridization_expected

    mapping_expected = [
        {"petabEntityId": "net1_input0", "modelEntityId": "net1.inputs[0][0]"},
        {"petabEntityId": "net1_input1", "modelEntityId": "net1.inputs[0][1]"},
        {"petabEntityId": "net1_output0", "modelEntityId": "net1.outputs[0][0]"},
        {"petabEntityId": "net1_output1", "modelEntityId": "net1.outputs[0][1]"},
        {"petabEntityId": "net1_ps", "modelEntityId": "net1.parameters"},
    ]
    with open(test_dir / "mapping.tsv", "r") as f:
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
    with open(test_dir / "parameters.tsv", "r") as f:
        parameters = list(csv.DictReader(f, delimiter="\t"))
        assert parameters == parameters_expected


def test_create_neural_ode_with_options(dir_tmp: Path):
    """create_neural_ode"""

    test_dir = dir_tmp / "petab"
    test_dir.mkdir()

    species = {"prey": 1.0, "predator": 2.0}
    create_neural_ode(
        species,
        model_filename="lv.xml",
        network_name="mynet",
        save_directory=test_dir,
    )

    expected_files = [
        "lv.xml",
        "hybridization.tsv",
        "mapping.tsv",
        "parameters.tsv",
    ]
    for fname in expected_files:
        assert (test_dir / fname).is_file()

    document = readSBML(test_dir / "lv.xml")
    model = document.getModel()
    species_list = model.getListOfSpecies()

    assert species_list.get("prey").getInitialAmount() == 1.0
    assert species_list.get("predator").getInitialAmount() == 2.0

    mapping_expected = [
        {"petabEntityId": "mynet_input0", "modelEntityId": "mynet.inputs[0][0]"},
        {"petabEntityId": "mynet_input1", "modelEntityId": "mynet.inputs[0][1]"},
        {"petabEntityId": "mynet_output0", "modelEntityId": "mynet.outputs[0][0]"},
        {"petabEntityId": "mynet_output1", "modelEntityId": "mynet.outputs[0][1]"},
        {"petabEntityId": "mynet_ps", "modelEntityId": "mynet.parameters"},
    ]
    with open(test_dir / "mapping.tsv", "r") as f:
        mapping = list(csv.DictReader(f, delimiter="\t"))
        assert mapping == mapping_expected


def test_create_neural_ode_problem(dir_tmp):
    """create_neural_ode_problem"""

    test_dir = dir_tmp / "petab"
    test_dir.mkdir()

    species = ["prey", "predator"]
    create_neural_ode(species, save_directory=test_dir)

    _write_test_measurements_file(test_dir)
    _write_test_network_file(test_dir)
    network_filename = "net1.yaml"
    measurements_filename = "measurements.tsv"
    observables_filename = "observables.tsv"
    array_filenames = ["supernet_params.h5", "supernet_inputs.h5"]

    create_neural_ode_problem(
        "model.xml",
        measurements_filename,
        observables_filename,
        network_filename,
        array_filenames,
        save_directory=test_dir,
    )

    expected_files = [
        "conditions.tsv",
        "problem.yaml",
    ]
    for fname in expected_files:
        assert (test_dir / fname).is_file()

    conditions_expected = [{"conditionId": "cond1"}]
    with open(test_dir / "conditions.tsv", "r") as f:
        condition = list(csv.DictReader(f, delimiter="\t"))
        assert condition == conditions_expected

    with open(test_dir / "problem.yaml", "r") as f:
        problem = safe_load(f)
        assert problem["model_files"]["model"]["location"] == "model.xml"
        assert problem["measurement_files"][0] == "measurements.tsv"

        assert "net1" in problem["extensions"]["sciml"]["neural_nets"]
        assert (
            problem["extensions"]["sciml"]["neural_nets"]["net1"]["location"]
            == "net1.yaml"
        )
        assert "array_files" in problem["extensions"]["sciml"]
        assert problem["extensions"]["sciml"]["array_files"] == array_filenames


def _write_test_measurements_file(test_dir: Path):
    """Write a simple measurements file for testing purposes."""
    measurements = [
        {
            "observableId": "prey_o",
            "simulationConditionId": "cond1",
            "time": "1.0",
            "measurement": "0.1",
        },
        {
            "observableId": "prey_o",
            "simulationConditionId": "cond1",
            "time": "2.0",
            "measurement": "0.5",
        },
        {
            "observableId": "predator_o",
            "simulationConditionId": "cond1",
            "time": "1.0",
            "measurement": "0.8",
        },
        {
            "observableId": "predator_o",
            "simulationConditionId": "cond1",
            "time": "1.0",
            "measurement": "0.2",
        },
    ]
    with open(test_dir / "measurements.tsv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=measurements[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(measurements)


def _write_test_network_file(test_dir: Path):
    """Write a simple network yaml file for testing purposes."""

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(2, 5)
            self.layer2 = torch.nn.Linear(5, 5)
            self.layer3 = torch.nn.Linear(5, 2)

        def forward(self, net_input):
            x = self.layer1(net_input)
            x = F.tanh(x)
            x = self.layer2(x)
            x = F.tanh(x)
            x = self.layer3(x)
            return x

    net1 = NeuralNetwork()
    nn_model1 = NNModel.from_pytorch_module(
        module=net1, nn_model_id="net1", inputs=[Input(input_id="input0")]
    )
    NNModelStandard.save_data(data=nn_model1, filename=(test_dir / "net1.yaml"))
