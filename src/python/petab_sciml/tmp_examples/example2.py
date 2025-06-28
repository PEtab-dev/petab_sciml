import torch
import torch.nn as nn
import torch.nn.functional as F
from petab_sciml.standard.nn_model import Input, NNModel, NNModelStandard


class Net(nn.Module):
    """Single layer."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define the computational graph."""
        x = self.conv1(input)
        x = F.relu(x)
        return x


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net()
nn_model0 = NNModel.from_pytorch_module(
    module=net0, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model0, filename="data2/nn_model0.yaml"
)

# Read the stored model from disk, reconstruct the pytorch module
loaded_model = NNModelStandard.load_data("data2/nn_model0.yaml")
net1 = loaded_model.to_pytorch_module()

print(net1.code)  # noqa: T201

# Store the pytorch module to disk again and verify that the round-trip was successful
nn_model1 = NNModel.from_pytorch_module(
    module=net1, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model1, filename="data2/nn_model1.yaml"
)

with open("data2/nn_model0.yaml") as f:
    data0 = f.read()
with open("data2/nn_model1.yaml") as f:
    data1 = f.read()


if not data0 == data1:
    raise ValueError(
        "The round-trip of saving the pytorch modules to disk failed."
    )
