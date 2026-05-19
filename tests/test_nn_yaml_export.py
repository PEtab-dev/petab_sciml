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
    """Example network with LayerNorm and tuple argument."""

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
    """Example network with LayerNorm and tuple argument."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 10)  # 5*5 from image dimension
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Execute the computational graph."""
        x = F.tanh(self.fc1(input))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Net5(nn.Module):
    """Network with `torch.cat` and multiple inputs and outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input = torch.cat([input1, input2])

        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output1 = f1
        output2 = f2
        return output1, output2


def test_nn_model_yaml_round_trip(dir_tmp: Path):
    """Test PyTorch module to PEtab-SciML YAML to PyTorch module round-trip."""
    filename_original = dir_tmp / "nn_model_original.yaml"
    filename_loaded = dir_tmp / "nn_model_loaded.yaml"

    _test_roundtrip(filename_original, filename_loaded, Net1())
    _test_roundtrip(filename_original, filename_loaded, Net2())
    _test_roundtrip(filename_original, filename_loaded, Net3())
    _test_roundtrip(filename_original, filename_loaded, Net4())
    _test_roundtrip(filename_original, filename_loaded, Net5())


def _test_roundtrip(filename_original, filename_loaded, net_module):
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
    loaded_model = NNModelStandard.load_data(filename_original)
    loaded_pytorch_module = loaded_model.to_pytorch_module()

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
