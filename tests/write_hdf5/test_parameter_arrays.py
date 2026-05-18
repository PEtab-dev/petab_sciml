import os

import torch
import torch.nn.functional as F
import numpy as np
import h5py
from torch import nn

from petab_sciml.standard.array_data import (
    ArrayData,
    ArrayDataStandard,
    extract_torch_parameters,
    extract_nn_yaml_parameters,
)
from petab_sciml.standard.nn_model import NNModel, NNModelStandard


class NetTest1(nn.Module):
    """ "Module with all PEtab-SciML supported layers"""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 5)
        self.layer2 = nn.Bilinear(5, 10, 2)
        self.layer3 = nn.Conv1d(1, 2, 5)
        self.layer4 = nn.Conv2d(2, 1, (5, 2))
        self.layer5 = nn.Conv3d(2, 1, (5, 4, 3))
        self.layer6 = nn.ConvTranspose1d(2, 1, 5)
        self.layer7 = nn.ConvTranspose2d(2, 1, (5, 2))
        self.layer8 = nn.ConvTranspose3d(2, 1, (5, 4, 3))
        self.layer9 = nn.MaxPool3d((3, 2, 1))
        self.layer10 = nn.Flatten()
        self.layer11 = nn.AvgPool3d((3, 2, 1))
        self.layer12 = nn.LPPool3d(2, (3, 2, 1))
        self.layer13 = nn.AdaptiveMaxPool3d((3, 2, 1))
        self.layer14 = nn.AdaptiveAvgPool3d((3, 2, 1))
        self.layer15 = nn.Dropout(0.5)
        self.layer16 = nn.AlphaDropout(0.5)
        self.layer17 = nn.Dropout2d(0.5)
        self.layer18 = nn.LayerNorm((4, 10, 11, 12))

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        return net_input


class NetTest2(nn.Module):
    """ "Testing writing parameters from YAML"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output


def test_write_parameters_torch(dir_tmp):
    """Test writing all named PyTorch parameters grouped by layer and name."""
    file1 = os.path.join(dir_tmp, "file1.hdf5")

    net_test = NetTest1()

    net_ps = extract_torch_parameters(net_test, "net1")
    array_data = ArrayData.model_validate(net_ps)
    ArrayDataStandard.save_data(array_data, file1)

    with h5py.File(file1, "r") as hdf5_file:
        assert "metadata" in hdf5_file
        assert "pytorch_format" in hdf5_file["metadata"]
        assert bool(hdf5_file["metadata"]["pytorch_format"][()]) is True

        assert "parameters" in hdf5_file
        assert "net1" in hdf5_file["parameters"]

        # Test values for each layer
        for name, parameter in net_test.named_parameters():
            layer_id, parameter_id = name.rsplit(".", maxsplit=1)

            assert layer_id in hdf5_file["parameters"]["net1"]
            assert parameter_id in hdf5_file["parameters"]["net1"][layer_id]

            written_data = hdf5_file["parameters"]["net1"][layer_id][parameter_id][()]
            expected_data = parameter.detach().cpu().numpy()
            np.testing.assert_array_equal(written_data, expected_data)


def test_write_parameters_nn_yaml(dir_tmp):
    """ "Test writing parameters for NN YAML file"""
    path_yaml = os.path.join(dir_tmp, "file1.yaml")
    path_hdf5 = os.path.join(dir_tmp, "file1.hdf5")

    torch.manual_seed(1)
    net_test = NetTest2()
    net_export = NNModel.from_pytorch_module(module=net_test, nn_model_id="net1")
    NNModelStandard.save_data(data=net_export, filename=path_yaml)

    torch.manual_seed(1)
    net_ps = extract_nn_yaml_parameters(path_yaml)
    array_data = ArrayData.model_validate(net_ps)
    ArrayDataStandard.save_data(array_data, path_hdf5)

    with h5py.File(path_hdf5, "r") as hdf5_file:
        assert "metadata" in hdf5_file
        assert "pytorch_format" in hdf5_file["metadata"]
        assert bool(hdf5_file["metadata"]["pytorch_format"][()]) is True

        assert "parameters" in hdf5_file
        assert "net1" in hdf5_file["parameters"]

        # Test values for each layer
        for name, parameter in net_test.named_parameters():
            layer_id, parameter_id = name.rsplit(".", maxsplit=1)

            assert layer_id in hdf5_file["parameters"]["net1"]
            assert parameter_id in hdf5_file["parameters"]["net1"][layer_id]

            written_data = hdf5_file["parameters"]["net1"][layer_id][parameter_id][()]
            expected_data = parameter.detach().cpu().numpy()
            np.testing.assert_array_equal(written_data, expected_data)
