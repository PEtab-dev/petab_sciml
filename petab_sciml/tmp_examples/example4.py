import torch
import torch.nn as nn
from petab_sciml.standard import Input, NNModel, NNModelStandard


class Net1(nn.Module):
    """Example network with BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.RNN(5, 10, 3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Define the computational graph."""
        x = self.layer1(input)
        return x


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
net0 = Net1()
input_ = torch.ones(1, 5, 5)
net0.forward(input_)
nn_model0 = NNModel.from_pytorch_module(
    module=net0, nn_model_id="model0", inputs=[Input(input_id="input0")]
)
NNModelStandard.save_data(
    data=nn_model0, filename="data4/nn_model0.yaml"
)
