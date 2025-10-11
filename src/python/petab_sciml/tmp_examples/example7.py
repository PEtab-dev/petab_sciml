import torch
import torch.nn as nn
import torch.nn.functional as F
from petab_sciml.standard.nn_model import Input, NNModel, NNModelStandard


class Net(nn.Module):
    """Example network.

    Ref: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        #input = torch.cat(input1, input2)
        input = torch.cat([input1, input2])

        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output1 = f1
        #return output1
        output2 = f2
        return output1, output2


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net()
nn_model0 = NNModel.from_pytorch_module(
    module=net0, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model0, filename="data7/nn_model0.yaml"
)

# Read the stored model from disk, reconstruct the pytorch module
loaded_model = NNModelStandard.load_data("data7/nn_model0.yaml")
net1 = loaded_model.to_pytorch_module()

# Store the pytorch module to disk again and verify that the round-trip was successful
nn_model1 = NNModel.from_pytorch_module(
    module=net1, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model1, filename="data7/nn_model1.yaml"
)

with open("data7/nn_model0.yaml") as f:
    data0 = f.read()
with open("data7/nn_model1.yaml") as f:
    data1 = f.read()


if not data0 == data1:
    raise ValueError(
        "The round-trip of saving the pytorch modules to disk failed."
    )
