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
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define the computational graph."""
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net()
nn_model0 = NNModel.from_pytorch_module(
    module=net0, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model0, filename="data/nn_model0.yaml"
)

# Read the stored model from disk, reconstruct the pytorch module
loaded_model = NNModelStandard.load_data("data/nn_model0.yaml")
net1 = loaded_model.to_pytorch_module()

# Store the pytorch module to disk again and verify that the round-trip was successful
nn_model1 = NNModel.from_pytorch_module(
    module=net1, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model1, filename="data/nn_model1.yaml"
)

with open("data/nn_model0.yaml") as f:
    data0 = f.read()
with open("data/nn_model1.yaml") as f:
    data1 = f.read()


if not data0 == data1:
    raise ValueError(
        "The round-trip of saving the pytorch modules to disk failed."
    )
