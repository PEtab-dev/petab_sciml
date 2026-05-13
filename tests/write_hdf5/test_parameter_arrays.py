import os
import shutil
import tempfile
import pytest

import torch
import numpy as np
import h5py
from torch import nn

from petab_sciml.hdf5.write_hdf5 import write_parameter_hdf5


class NetTest(nn.Module):
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


@pytest.fixture
def dir_tmp():
    """Create and remove a temporary directory for each test."""
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


def test_write_parameters(dir_tmp):
    """Test writing all named PyTorch parameters grouped by layer and name."""
    file1 = os.path.join(dir_tmp, "file1.hdf5")
    net_test = NetTest()

    file1 = write_parameter_hdf5(file1, net_test, "net1")
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


def test_write_over(dir_tmp):
    """Test raising on existing parameter datasets and explicit overwriting."""
    file1 = os.path.join(dir_tmp, "file1.hdf5")
    net_test = NetTest()
    file1 = write_parameter_hdf5(file1, net_test, "net1")

    with pytest.raises(ValueError, match="already exists"):
        write_parameter_hdf5(file1, net_test, "net1")

    # Test can over-write
    file1 = write_parameter_hdf5(file1, net_test, "net1", on_dataset_exists="overwrite")
    assert os.path.isfile(file1)
