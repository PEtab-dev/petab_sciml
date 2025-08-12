import torch
import torch.nn as nn
import torch.nn.functional as F
from petab_sciml.standard import Input, NNModel, NNModelStandard


class Net(nn.Module):
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


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net()

nn_model0 = NNModel.from_pytorch_module(
    module=net0, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model0, filename="data6/nn_model0.yaml"
)
