from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type
from numpy.typing import ArrayLike
from pydantic import BaseModel, field_validator
from ruamel.yaml import YAML

if TYPE_CHECKING:
    import torch


__all__ = [
    "Metadata",
    "ArrayData",
    "ArrayDataStandard",
    "METADATA",
    "DATA",
    "CONDITION_IDS",
    "INPUTS",
    "PARAMETERS",
    "ALL_CONDITION_IDS",
    "extract_torch_parameters",
    "add_array_files_to_yaml",
]


METADATA = "metadata"
DATA = "data"
CONDITION_IDS = "conditionIds"
INPUTS = "inputs"
PARAMETERS = "parameters"
ALL_CONDITION_IDS = "0"


Array = get_array_type()


class Metadata(BaseModel):
    """Metadata for array(s)."""

    pytorch_format: bool
    """Whether the arrays match the default PyTorch format.

    For example, PyTorch uses row-major arrays, and the weight matrix
    of its "Linear layer" is `out_features x in_features`.

    True indicates that the arrays can be directly used in PyTorch.
    False indicates that some array operations are first required.
    """


class ArrayData(BaseModel):
    """Multiple arrays.

    For example, data for different inputs for different conditions,
    or values for different parameters of different layers.
    """

    metadata: Metadata
    """Additional metadata for the arrays."""

    inputs: dict[str, dict[str, Array]] = {}
    """Input data arrays.

    Outer keys are input IDs.
    Inner dict keys are semicolon-delimited lists of condition IDs,
    and values are the corresponding input array data for those conditions.
    """

    parameters: dict[str, dict[str, dict[str, Array]]] = {}
    """Parameter value arrays.

    Outer dict keys are NN model IDs.
    Inner dict keys are layer IDs.
    Inner inner dict keys are layer-specific parameter IDs, and values are the
    corresponding array data.
    """

    @field_validator(INPUTS, mode="after")
    @classmethod
    def validate_condition_ids(cls, inputs) -> dict[str, ArrayLike]:
        for input_id, input_data in inputs.items():
            if not input_data:
                raise ValueError(f"No input data supplied for input `{input_id}`.")

            for condition_ids_str, array in input_data.items():
                n_arrays = len(input_data)
                if (condition_ids_str == ALL_CONDITION_IDS) and (n_arrays != 1):
                    raise ValueError(
                        "The condition IDs list is "
                        f"`{ALL_CONDITION_IDS}`, which indicates that the "
                        "array will be applied to all conditions. In this "
                        "case, exactly one array must be specified. "
                        f"However, {n_arrays} arrays were specified for "
                        f"input `{input_id}`."
                    )
        return inputs


def extract_torch_parameters(torch_module: "torch.nn.Module", nn_model_id: str) -> dict:
    """Extract parameters as NumPy arrays from a PyTorch module.

    Parameters are grouped by layer ID and parameter ID using the final dot
    separator in each named parameter, e.g. `layer.weight` or `layer.bias`.

    Args:
        torch_module: A PyTorch module whose `named_parameters()` will be
            converted to NumPy arrays.
        nn_model_id:
            Neural network model ID, as defined in the PEtab-SciML YAML file.

    Returns:
        A nested dictionary compatible with ArrayData for exporting to
        PEtab-SciML HDF5-file format
    """
    array_dict = {METADATA: {"pytorch_format": True}}
    parameters_dict = array_dict.setdefault(PARAMETERS, {})
    parameters_net_dict = parameters_dict.setdefault(nn_model_id, {})

    for name, value in torch_module.named_parameters():
        # Layer with no parameters to estimate
        if value.numel() == 0:
            continue

        try:
            layer_id, parameter_id = name.rsplit(".", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Expected PyTorch parameter name of the form "
                f"'<layer_id>.<parameter_id>', got {name!r}."
            ) from exc

        array = _to_numpy_array(value)
        parameters_net_dict.setdefault(layer_id, {})[parameter_id] = array

    return array_dict


def add_array_files_to_yaml(
    yaml_file: str,
    array_files: str | Iterable[str],
    on_existing: Literal["ignore", "raise"] = "ignore",
) -> str:
    """Add PEtab-SciML HDF5 array file(s) to a PEtab problem YAML file.

    Args:
        yaml_file:
            Path to the PEtab problem YAML file to update in place.
        array_files:
            Array file path or array file paths to add to the YAML file. Files
            must be located in the same directory as ``yaml_file`` and are
            stored by file name only.
        on_existing:
            How to handle array files that are already listed.
            - ``"ignore"``: keep the existing entry and do not add a duplicate.
            - ``"raise"``: raise an error if an array file is already listed.

    Returns:
        str:
            Path to the updated YAML file.
    """
    if on_existing not in {"ignore", "raise"}:
        raise ValueError("on_existing must be either 'ignore' or 'raise'.")

    if isinstance(array_files, str):
        array_files = [array_files]

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_file, "r") as f:
        data = yaml.load(f)

    yaml_dir = os.path.abspath(os.path.dirname(yaml_file) or ".")

    extensions = data.setdefault("extensions", {})
    petab_sciml = extensions.setdefault("petab_sciml", {})
    existing_array_files = petab_sciml.setdefault("array_files", [])

    for array_file in array_files:
        array_dir = os.path.abspath(os.path.dirname(array_file))
        if array_dir != yaml_dir:
            raise ValueError(
                "Array files must be located in the same directory as the "
                "YAML file. Got array file "
                f"{array_file!r}, but YAML directory is {yaml_dir!r}."
            )

        array_file_name = os.path.basename(array_file)
        if array_file_name in existing_array_files:
            if on_existing == "raise":
                raise ValueError(
                    f"Array file {array_file_name!r} is already listed in "
                    "'extensions.petab_sciml.array_files'."
                )
            continue

        existing_array_files.append(array_file_name)

    with open(yaml_file, "w") as f:
        yaml.dump(data, f)

    return yaml_file


def _to_numpy_array(array: ArrayLike) -> np.ndarray:
    """Convert supported array-like objects to data accepted by h5py."""
    if hasattr(array, "detach"):
        array = array.detach().numpy()
    else:
        try:
            array = np.asarray(array)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Input array of type {type(array).__name__} could not be "
                "converted to a NumPy array."
            ) from exc
    return array


ArrayDataStandard = Hdf5Standard(model=ArrayData)


if __name__ == "__main__":
    from pathlib import Path

    ArrayDataStandard.save_schema(
        Path(__file__).resolve().parents[4]
        / "doc"
        / "standard"
        / "array_data_schema.json"
    )
