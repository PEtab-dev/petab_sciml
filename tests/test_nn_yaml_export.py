from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from petab_sciml.standard.nn_model import Input, NNModel, NNModelStandard


class Net1(nn.Module):
    """Example convolutional neural network."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define the computational graph."""
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output


class Net2(nn.Module):
    """Single layer."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define the computational graph."""
        x = self.conv1(input)
        x = F.relu(x)
        return x


class Net3(nn.Module):
    """Example network with LayerNormt."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm((4, 10, 11, 12))
        self.layer1 = nn.Conv3d(4, 1, 5)
        self.flatten1 = nn.Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute the computational graph."""
        x = self.norm1(input)
        x = self.layer1(x)
        x = self.flatten1(x)
        return x


class Net4(nn.Module):
    """Network with `torch.cat` and multiple inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input = torch.cat([input1, input2])

        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        return f2


def test_nn_model_yaml_round_trip(dir_tmp: Path):
    """Test PyTorch module to PEtab-SciML YAML to PyTorch module round-trip."""
    filename_original = dir_tmp / "nn_model_original.yaml"
    filename_loaded = dir_tmp / "nn_model_loaded.yaml"

    test_cases = [
        ("net1", Net1, lambda: torch.rand(1, 1, 32, 32)),
        ("net2", Net2, lambda: torch.rand(1, 1, 32, 32)),
        ("net3", Net3, lambda: torch.rand(1, 4, 10, 11, 12)),
        ("net4", Net4, lambda: (torch.rand(60), torch.rand(60))),
    ]

    for name, net_cls, make_input in test_cases:
        torch.manual_seed(1)
        _test_roundtrip(filename_original, filename_loaded, net_cls(), make_input())


def _test_roundtrip(filename_original, filename_loaded, net_module, torch_inputs):
    """Helper function to test roundtrip"""
    # Convert PyTorch module to PEtab-SciML model and save to YAML.
    nn_model_original = NNModel.from_pytorch_module(
        module=net_module,
        nn_model_id="model0",
        inputs=[Input(input_id="input0")],
    )
    NNModelStandard.save_data(
        data=nn_model_original,
        filename=filename_original,
    )

    # Load YAML and reconstruct the PyTorch module.
    torch.manual_seed(1)
    loaded_model = NNModelStandard.load_data(filename_original)
    loaded_pytorch_module = loaded_model.to_pytorch_module()

    # Check that original and reconstructed modules have consistent output
    _assert_same_output(net_module, loaded_pytorch_module, torch_inputs)

    # Convert the reconstructed PyTorch module back to PEtab-SciML and save.
    nn_model_loaded = NNModel.from_pytorch_module(
        module=loaded_pytorch_module,
        nn_model_id="model0",
        inputs=[Input(input_id="input0")],
    )
    NNModelStandard.save_data(
        data=nn_model_loaded,
        filename=filename_loaded,
    )

    # Compare the generated YAML files.
    with open(filename_original) as f:
        data_original = f.read()

    with open(filename_loaded) as f:
        data_loaded = f.read()

    assert data_original == data_loaded


def _assert_same_output(module_a, module_b, torch_inputs):
    """Assert that two PyTorch modules give the same output for fixed inputs."""
    if isinstance(torch_inputs, tuple):
        output_a = module_a.forward(torch_inputs[0], torch_inputs[1])
        output_b = module_b.forward(torch_inputs[0], torch_inputs[1])
    else:
        output_a = module_a.forward(torch_inputs)
        output_b = module_b.forward(torch_inputs)

    assert torch.equal(output_a, output_b)
