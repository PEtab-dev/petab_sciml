# ruff: noqa: F401, F821, F841
import equinox as eqx
import jax.nn
import jax.random as jr
import jax
import amici.jax.nn


class net3(eqx.Module):
    layers: dict
    inputs: list[str]
    outputs: list[str]

    def __init__(self, key):
        super().__init__()
        keys = jr.split(key, 3)
        self.layers = {'layer1': eqx.nn.Conv2d(stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, padding_mode='zeros', in_channels=3, out_channels=1, kernel_size=[5, 5], use_bias=True, key=keys[0]),
            'layer2': amici.jax.nn.Flatten(start_dim=1, end_dim=-1),
            'layer3': eqx.nn.Linear(in_features=36, out_features=1, use_bias=True, key=keys[2])}
        self.inputs = ['input0']
        self.outputs = ['relu']

    def forward(self, input, key=None):
        net_input = input
        layer1 = (jax.vmap(self.layers['layer1']) if len(net_input.shape) == 4 else self.layers['layer1'])(net_input, )
        layer2 = self.layers['layer2'](layer1, )
        layer3 = (jax.vmap(self.layers['layer3']) if len(layer2.shape) == 2 else self.layers['layer3'])(layer2, )
        relu = jax.nn.relu(layer3, )
        output = relu
        return output


net = net3
