from contextlib import chdir
import csv
from petab_sciml.problem_utils.neural_ode import (
    create_neural_ode, 
    create_neural_ode_problem,
)
from petab_sciml.standard.nn_model import NNModelStandard, NNModel, Input
import os
import torch
from torch import nn
import torch.nn.functional as F
from yaml import safe_load
from libsbml import readSBML


def test_create_neural_ode(tmp_path):
    """create_neural_ode"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        species = ["prey", "predator"]
        create_neural_ode(species)

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

def test_create_neural_ode_with_options(tmp_path):
    """create_neural_ode"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        os.mkdir("petab_problem")
        species = {"prey": 1.0, "predator": 2.0}
        create_neural_ode(
            species, 
            model_filename="lv.xml", 
            network_name="mynet",
            save_directory="./petab_problem"
        )

        expected_files = [
            "petab_problem/lv.xml",
            "petab_problem/hybridization.tsv",
            "petab_problem/mapping.tsv",
            "petab_problem/parameters.tsv",
        ]
        for fname in expected_files:
            assert os.path.isfile(fname)
            
        document = readSBML("petab_problem/lv.xml")
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
        with open("petab_problem/mapping.tsv", "r") as f:
            mapping = list(csv.DictReader(f, delimiter="\t"))
            assert mapping == mapping_expected


def test_create_neural_ode_problem(tmp_path):
    """create_neural_ode_problem"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        species = ["prey", "predator"]
        create_neural_ode(species)

        _write_test_measurements_file()
        _write_test_network_file()
        network_filename = "net1.yaml"
        measurements_filename = "measurements.tsv"
        observables_filename = "observables.tsv"
        array_filenames = ["supernet_params.h5", "supernet_inputs.h5"]

        create_neural_ode_problem(
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

def _write_test_network_file():
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
    NNModelStandard.save_data(
        data=nn_model1, filename="net1.yaml"
    )