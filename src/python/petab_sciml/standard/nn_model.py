from __future__ import annotations

import ast
import inspect
import re

import torch.fx
import torch.nn as nn
from pydantic import BaseModel, Field

from mkstd import YamlStandard


__all__ = ["Input", "Layer", "Node", "NNModel", "NNModelStandard"]


class Input(BaseModel):
    """Specify (transformations of) the input layer."""

    input_id: str


class Layer(BaseModel):
    """Specify layers."""

    layer_id: str
    layer_type: str
    # FIXME currently handled as kwargs
    args: dict | None = Field(
        default=None
    )  # TODO class of layer-specific supported args


class Node(BaseModel):
    """A node of the computational graph.

    e.g. a node in the forward call of a PyTorch model.
    Ref: https://pytorch.org/docs/stable/fx.html#torch.fx.Node
    """

    name: str
    op: str
    target: str
    args: list | None = Field(default=None)
    kwargs: dict | None = Field(default=None)


# Some customization needs to be manually extracted. These are things
# that might only be present in `Module.__repr__` output conditionally,
# usually because pytorch avoids outputting default values in its `__repr__`.
# We will explicitly save them so tools don't need to find out what pytorch
# defaults are.
extra_repr = {
    "Conv2d": {
        # Missing from `Conv2d.__init__`: `output_padding`
        # "output_padding": lambda m: m.output_padding,
        "__all__": ["padding", "dilation", "groups", "bias", "padding_mode"],
        "getters": {
            "bias": lambda m: m.bias is not None,
        },
    },
    "RNN": {
        "__all__": [
            "input_size",
            "hidden_size",
            "proj_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
        ],
        "getters": {},
    },
}

extra_repr = {
    module_id: {
        attr: module_def["getters"].get(
            attr, lambda m, attr=attr: getattr(m, attr)
        )
        for attr in module_def["__all__"]
    }
    for module_id, module_def in extra_repr.items()
}


def extract_module_args(module: nn.Module) -> dict:
    """Get the arguments used to create the module.

    N.B.: currently, all arguments must be Python literals compatible
    with `ast.literal_eval`, and cannot contain the character `=`.

    Args:
        module:
            The model.

    Returns:
        The arguments, as keyword arguments.
    """
    # Stage 1: get all arguments in the intersection of `__constants__` and
    # `__init__`
    init_arg_names = set(inspect.signature(module.__init__).parameters)
    constant_init_args = {
        arg: getattr(module, arg)
        for arg in module.__constants__
        if arg in init_arg_names
    }

    # Stage 2: add all arguments suggested in the `__repr__
    ## Get names of module positional arguments
    arg_names = []
    for arg in inspect.signature(module.__init__).parameters.values():
        if arg.name == "self":
            continue
        if arg.default != inspect.Parameter.empty:
            continue
        if arg.kind not in [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ]:
            continue
        arg_names.append(arg.name)

    ## Extract all arguments
    class_name, all_args_str = re.match(
        r"(\w+)\((.*)\)", repr(module).strip()
    ).groups()

    ## All positional arguments exist
    args = {}
    args_str_list = []
    for arg_str in all_args_str.split(","):
        if "=" in arg_str:
            break
        args_str_list.append(arg_str)
    if args_str_list:
        args = dict(
            zip(arg_names, ast.literal_eval(",".join(args_str_list) + ","))
        )

    kwargs = {
        kw: ast.literal_eval(arg_str.strip())
        for kw, arg_str in re.findall(
            r"(\w*)=([^=]*)(?=,\s*\w+=|$)", all_args_str
        )
    }

    # See `extra_repr`
    extra = {
        kw: getter(module)
        for kw, getter in extra_repr.get(
            get_module_layer_type(module), {}
        ).items()
    }

    return constant_init_args | extra | args | kwargs


def get_module_layer_type(module: nn.Module) -> str:
    """Get the layer type of a module.

    Args:
        module:
            The module.

    Returns:
        The module type.
    """
    return type(module).__name__


class NNModel(BaseModel):
    """An easy-to-use format to specify simple deep NN models.

    There is a function to export this to a PyTorch module, or to YAML.
    """

    nn_model_id: str

    inputs: list[Input]

    layers: list[Layer]
    """The components of the model (e.g., layers of a neural network)."""

    forward: list[Node]

    @staticmethod
    def from_pytorch_module(
        module: nn.Module, nn_model_id: str, inputs: list[Input]
    ) -> NNModel:
        """Create a PEtab SciML NN model from a pytorch module."""
        layers = []
        layer_ids = []
        for layer_id, layer_module in module.named_modules():
            if not layer_id:
                # first entry is all modules combined
                continue
            layer = Layer(
                layer_id=layer_id,
                layer_type=get_module_layer_type(layer_module),
                args=extract_module_args(module=layer_module),
            )
            layers.append(layer)
            layer_ids.append(layer_id)

        nodes = []
        node_names = []
        pytorch_nodes = list(torch.fx.symbolic_trace(module).graph.nodes)
        for pytorch_node in pytorch_nodes:
            op = pytorch_node.op
            target = pytorch_node.target
            if op == "call_function":
                target = pytorch_node.target.__name__
            node = Node(
                name=pytorch_node.name,
                op=pytorch_node.op,
                target=target,
                args=[
                    (arg if arg not in pytorch_nodes else str(arg))
                    for arg in pytorch_node.args
                ],
                kwargs=pytorch_node.kwargs,
            )
            nodes.append(node)
            node_names.append(node.name)

        nn_model = NNModel(
            nn_model_id=nn_model_id, inputs=inputs, layers=layers, forward=nodes
        )
        return nn_model

    def to_pytorch_module(self) -> nn.Module:
        """Create a pytorch module from a PEtab SciML NN model."""
        self2 = self

        class _PytorchModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for layer in self2.layers:
                    setattr(
                        self,
                        layer.layer_id,
                        getattr(nn, layer.layer_type)(**layer.args),
                    )

        graph = torch.fx.Graph()
        state = {}
        for node in self.forward:
            args = []
            if node.args:
                for node_arg in node.args:
                    arg = node_arg
                    try:
                        if arg in state:
                            arg = state[arg]
                    except TypeError:
                        pass
                    args.append(arg)
            args = tuple(args)
            kwargs = {}
            if node.kwargs:
                kwargs = {k: state.get(v, v) for k, v in node.kwargs.items()}
            match node.op:
                case "placeholder":
                    state[node.name] = graph.placeholder(node.target)
                case "call_function":
                    if node.target in ["flatten"]:
                        function = getattr(torch, node.target)
                    else:
                        function = getattr(nn.functional, node.target)
                    state[node.name] = graph.call_function(
                        function, args, kwargs
                    )
                case "call_module":
                    state[node.name] = graph.call_module(
                        node.target, args, kwargs
                    )
                case "output":
                    graph.output(args[0])

        return torch.fx.GraphModule(_PytorchModule(), graph)


NNModelStandard = YamlStandard(model=NNModel)


if __name__ == "__main__":
    from pathlib import Path

    from mkstd import JsonStandard


    # The NN model file is YAML-formatted, so the schema is provided in YAML format.
    NNModelStandard.save_schema(Path(__file__).resolve().parents[4] / "docs" / "src" / "assets" / "nn_model_schema.yaml")

    # However, the schema format is JsonSchema, so the schema is also provided redundantly in JSON schema.
    NNModelStandardJson = JsonStandard(model=NNModel)
    NNModelStandardJson.save_schema(Path(__file__).resolve().parents[4] / "docs" / "src" / "assets" / "nn_model_schema.json")
