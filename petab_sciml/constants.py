"""Constants for reserved keywords or useful lists."""

from __future__ import annotations

import re
from enum import StrEnum

__all__ = [
    "METADATA",
    "DATA",
    "CONDITION_IDS",
    "INPUTS",
    "PARAMETERS",
    "ALL_CONDITION_IDS",
    "ARRAY",
    "Layers",
    "ActivationFunctions",
    "TensorOps",
    "Op",
    "NN_ENTITY_PATTERN",
    "NN_PARAMETER_PATTERN",
]


# --- Array data file (HDF5) -------------------------------------------------

#: Metadata group.
METADATA = "metadata"
#: Name of a dataset.
DATA = "data"
#: Condition IDs field.
CONDITION_IDS = "conditionIds"
#: Inputs group.
INPUTS = "inputs"
#: Parameters group.
PARAMETERS = "parameters"
#: Condition IDs field value when the array data applies to all conditions.
ALL_CONDITION_IDS = "0"


# --- Reserved keywords ------------------------------------------------------

#: Indicate that a variable value is an array.
ARRAY = "array"


# --- NN model YAML format ---------------------------------------------------


class Layers(StrEnum):
    """Neural network layers supported by the NN model YAML format.

    Names follow the PyTorch (``torch.nn``) naming scheme.
    See https://petab-sciml.readthedocs.io/latest/layers.html
    """

    Linear = "Linear"
    Bilinear = "Bilinear"
    Flatten = "Flatten"
    Dropout = "Dropout"
    Dropout1d = "Dropout1d"
    Dropout2d = "Dropout2d"
    Dropout3d = "Dropout3d"
    AlphaDropout = "AlphaDropout"
    Conv1d = "Conv1d"
    Conv2d = "Conv2d"
    Conv3d = "Conv3d"
    ConvTranspose1d = "ConvTranspose1d"
    ConvTranspose2d = "ConvTranspose2d"
    ConvTranspose3d = "ConvTranspose3d"
    MaxPool1d = "MaxPool1d"
    MaxPool2d = "MaxPool2d"
    MaxPool3d = "MaxPool3d"
    AvgPool1d = "AvgPool1d"
    AvgPool2d = "AvgPool2d"
    AvgPool3d = "AvgPool3d"
    LPPool1d = "LPPool1d"
    LPPool2d = "LPPool2d"
    LPPool3d = "LPPool3d"
    AdaptiveMaxPool1d = "AdaptiveMaxPool1d"
    AdaptiveMaxPool2d = "AdaptiveMaxPool2d"
    AdaptiveMaxPool3d = "AdaptiveMaxPool3d"
    AdaptiveAvgPool1d = "AdaptiveAvgPool1d"
    AdaptiveAvgPool2d = "AdaptiveAvgPool2d"
    AdaptiveAvgPool3d = "AdaptiveAvgPool3d"


class ActivationFunctions(StrEnum):
    """Activation functions supported by the NN model YAML format.

    Names follow the PyTorch (``torch.nn.functional``) naming scheme.
    See https://petab-sciml.readthedocs.io/latest/layers.html
    """

    relu = "relu"
    relu6 = "relu6"
    hardtanh = "hardtanh"
    hardswish = "hardswish"
    selu = "selu"
    leaky_relu = "leaky_relu"
    gelu = "gelu"
    tanhshrink = "tanhshrink"
    softsign = "softsign"
    softplus = "softplus"
    tanh = "tanh"
    sigmoid = "sigmoid"
    hardsigmoid = "hardsigmoid"
    silu = "silu"
    mish = "mish"
    elu = "elu"
    celu = "celu"
    softmax = "softmax"
    log_softmax = "log_softmax"


class TensorOps(StrEnum):
    """Non-activation tensor operations supported in a forward pass.

    Unlike layers and activation functions, these are in ``torch``
    rather than ``torch.nn``/``torch.nn.functional``.
    """

    flatten = "flatten"
    cat = "cat"


class Op(StrEnum):
    """PyTorch ``torch.fx`` opcodes for the ``op`` field of a node in a
    neural network's forward graph.

    See https://pytorch.org/docs/stable/fx.html#torch.fx.Node
    """

    #: A forward-graph input (function argument).
    PLACEHOLDER = "placeholder"
    #: A call to a free function (e.g. a ``torch.nn.functional`` activation).
    CALL_FUNCTION = "call_function"
    #: A call to a method on a value (e.g. ``x.tanh()``).
    CALL_METHOD = "call_method"
    #: A call to a submodule/layer (e.g. a ``torch.nn`` layer).
    CALL_MODULE = "call_module"
    #: The forward-graph output.
    OUTPUT = "output"


# --- Mapping-table modelEntityId syntax -------------------------------------

#: Matches a mapping-table ``modelEntityId`` that refers to a neural network
#: input, output, or parameter, e.g. ``net1.inputs[0][1]`` or
#: ``net1.parameters[layer1]``. Named groups: ``nn_id``, ``entity_type``.
NN_ENTITY_PATTERN = re.compile(
    r"^(?P<nn_id>[^.\[\]\s]+)\.(?P<entity_type>inputs|outputs|parameters)\b"
)

#: Matches a mapping-table ``modelEntityId`` referring to neural network
#: parameters, optionally for a specific layer, e.g. ``net1.parameters`` or
#: ``net1.parameters[layer1]``. Named groups: ``nn_id``, ``layer``.
NN_PARAMETER_PATTERN = re.compile(
    r"^(?P<nn_id>[^.\[\]\s]+)\.parameters(?:\[(?P<layer>[^\]]+)\])?"
)
