from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type
from numpy.typing import ArrayLike
from pydantic import BaseModel, field_validator
from ruamel.yaml import YAML

from petab_sciml.standard.nn_model import NNModelStandard

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
    array_dict = {
        METADATA: {"pytorch_format": True},
        PARAMETERS: {
            nn_model_id: {},
        },
    }
    parameters_net_dict = array_dict[PARAMETERS][nn_model_id]

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

        val_as_array = _to_numpy_array(value)
        parameters_net_dict.setdefault(layer_id, {})[parameter_id] = val_as_array

    return array_dict


def extract_nn_yaml_parameters(yaml_file: str | Path) -> dict:
    """Extract parameters as NumPy arrays from a PEtab-SciML YAML file

    This function loads a PEtab-SciML neural-network YAML file, reconstructs
    the corresponding PyTorch module, and extracts the initialized module
    parameters as NumPy arrays that can be exported to the PEtab-SciML HDF5
    array-file format.

    Args:
        yaml_file:
            Path to a PEtab-SciML neural-network YAML file.

    Returns:
        A nested dictionary compatible with ArrayData for exporting to
        PEtab-SciML HDF5-file format
    """
    nn_model = NNModelStandard.load_data(yaml_file)
    torch_module = nn_model.to_pytorch_module()

    return extract_torch_parameters(torch_module, nn_model.nn_model_id)


def add_array_files_to_yaml(
    yaml_file: str | Path,
    array_files: str | Path | Iterable[str | Path],
    overwrite: bool = False,
) -> Path:
    """Add PEtab-SciML HDF5 array file(s) to a PEtab problem YAML file.

    Args:
        yaml_file:
            Path to the PEtab problem YAML file to update in place.
        array_files:
            Array file path or array file paths to add to the YAML file. Files
            may be given as absolute paths or as paths relative to the current
            working directory. They are stored in the YAML file as paths
            relative to the directory containing ``yaml_file``.
        overwrite:
            If ``True``, replace any existing list in
            ``extensions.petab_sciml.array_files`` with the provided files.
            If ``False``, append any files not already present (duplicates
            are ignored).

    Returns:
        Path: Path to the updated YAML file.
    """
    yaml_file = Path(yaml_file)

    if isinstance(array_files, (str, Path)):
        array_files = [Path(array_files)]
    else:
        array_files = [Path(array_file) for array_file in array_files]

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_file, "r") as f:
        data = yaml.load(f)

    yaml_dir = yaml_file.parent.resolve()

    extensions = data.setdefault("extensions", {})
    petab_sciml = extensions.setdefault("petab_sciml", {})
    existing_array_files = petab_sciml.setdefault("array_files", [])

    existing_array_file_paths = {
        # Handles both absolute and relative `array_file` correctly
        (yaml_dir / Path(array_file)).resolve()
        for array_file in existing_array_files
    }

    for array_file in array_files:
        array_file = array_file.resolve()

        if array_file in existing_array_file_paths:
            if overwrite is False:
                raise ValueError(
                    f"Array file {array_file.name!r} is already listed in "
                    "'extensions.petab_sciml.array_files'."
                )
            continue

        array_file_relative = array_file.relative_to(yaml_dir, walk_up=True).as_posix()
        existing_array_files.append(array_file_relative)

    with open(yaml_file, "w") as f:
        yaml.dump(data, f)

    return yaml_file


def _to_numpy_array(array: ArrayLike) -> np.ndarray:
    """Convert supported array-like objects to data accepted by h5py."""
    if hasattr(array, "detach"):
        array = array.detach().numpy()
    return np.asarray(array)


ArrayDataStandard = Hdf5Standard(model=ArrayData)


if __name__ == "__main__":
    from pathlib import Path

    ArrayDataStandard.save_schema(
        Path(__file__).resolve().parents[4]
        / "doc"
        / "standard"
        / "array_data_schema.json"
    )
