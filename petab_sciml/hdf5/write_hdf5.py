import os
import numpy as np
from typing import Iterable, Literal

import h5py
from numpy.typing import ArrayLike

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def write_input_hdf5(
    filename: str,
    input_id: str,
    arrays: ArrayLike | Iterable[ArrayLike],
    condition_ids: str | Iterable[str] | None = None,
    on_dataset_exists: Literal["raise", "overwrite"] = "raise",
    validate: bool = True,
) -> str:
    """Write PEtab-SciML input arrays to an HDF5 array file.

    Args:
    filename:
        Path to the HDF5 file. Created if it does not exist and appended to if
        it already exists.
    input_id:
        PEtab identifier that refers to an input of a neural network model.
    arrays:
        Input array or input arrays to write.

        If ``condition_ids`` is ``None``, ``arrays`` must be a single array
        that applies to all PEtab conditions.

        If ``condition_ids`` is provided, the input is condition-specific.
        For a single condition ID, ``arrays`` must be a single array. For
        multiple condition IDs, ``arrays`` must be an iterable of arrays with
        the same length as ``condition_ids``.
    condition_ids:
        Condition ID or condition IDs for condition-specific input arrays. If
        provided, must match the length of ``arrays``.
    on_dataset_exists:
        Handling target datasets that already exist.
        - ``"raise"``: raise an error.
        - ``"overwrite"``: delete and recreate the existing dataset.
    validate:
        Whether to validate the resulting HDF5 file structure with the
        PEtab-SciML linter

    Returns:
    Path:
        Path to the written HDF5 file.
    """
    if on_dataset_exists not in {"raise", "overwrite"}:
        raise ValueError("on_dataset_exists must be either 'raise' or 'overwrite'.")

    if condition_ids is None:
        condition_ids = "0"

    if not isinstance(condition_ids, str) and len(arrays) != len(condition_ids):
        raise ValueError(
            "The number of arrays must match the number of condition_ids. "
            f"Got {len(arrays)} arrays and {len(condition_ids)} "
            "condition IDs."
        )

    # First scenario is a single provided condition id
    if isinstance(condition_ids, str):
        datasets = [_to_hdf5_data(arrays)]
        condition_ids = [condition_ids]
    else:
        datasets = [_to_hdf5_data(x) for x in arrays]

    _ensure_parent_dir(filename)

    with h5py.File(filename, "a") as hdf5_file:
        _ensure_metadata(hdf5_file)

        input_group = hdf5_file.require_group(f"inputs/{input_id}")

        for condition_id, array in zip(condition_ids, datasets):
            if condition_id in input_group:
                if on_dataset_exists == "raise":
                    raise ValueError(
                        f"Dataset 'inputs/{input_id}/{condition_id}' "
                        "already exists. Set "
                        "on_dataset_exists='overwrite' to replace it."
                    )
                del input_group[condition_id]

            input_group.create_dataset(
                condition_id,
                data=array,
            )

    # TODO Validate file with PEtab-SciML linter!
    return filename


def write_parameter_hdf5(
    filename: str,
    torch_module: "torch.nn.Module",
    nn_model_id: str,
    on_dataset_exists: Literal["raise", "overwrite"] = "raise",
    validate: bool = True,
) -> str:
    """Write PyTorch parameters to PEtab-SciML HDF5 array file

    Args:
    filename:
        Path to the HDF5 file. Created if it does not exist and appended to if
        it already exists.
    source:
        PyTorch ``torch.nn.Module`` whose named parameters are written to the
        HDF5 file. Parameter names are expected to follow the PyTorch
        convention ``"<layer_id>.<parameter_id>"``, for example
        ``"layer1.weight"`` or ``"layer1.bias"``.
    nn_model_id:
        Neural network model ID, as defined in the PEtab-SciML YAML file.
    on_dataset_exists:
        Handling target datasets that already exist.
        - ``"raise"``: raise an error.
        - ``"overwrite"``: delete and recreate the existing dataset.
    validate:
        Whether to validate the resulting HDF5 file with the PEtab-SciML
        linter.

    Returns:
    Path:
        Path to the written HDF5 file.
    """
    if on_dataset_exists not in {"raise", "overwrite"}:
        raise ValueError("on_dataset_exists must be either 'raise' or 'overwrite'.")

    if not hasattr(torch_module, "named_parameters"):
        raise TypeError(
            "source must be a PyTorch torch.nn.Module with a named_parameters() method."
        )

    _ensure_parent_dir(filename)

    with h5py.File(filename, "a") as hdf5_file:
        _ensure_metadata(hdf5_file)

        parameters_dict = _np_arrays_from_torch_module(torch_module)
        for layer_id, layer_dict in parameters_dict.items():
            layer_group = hdf5_file.require_group(
                f"parameters/{nn_model_id}/{layer_id}"
            )

            for array_id, array in layer_dict.items():
                if array_id in layer_group:
                    if on_dataset_exists == "raise":
                        raise ValueError(
                            f"Dataset 'parameters/{nn_model_id}/{layer_id}/{array_id}' "
                            "already exists. Set "
                            "on_dataset_exists='overwrite' to replace it."
                        )
                    del layer_group[array_id]

                layer_group.create_dataset(array_id, data=array)

    # TODO Validate file with linter!
    return filename


def _to_hdf5_data(array: ArrayLike) -> np.ndarray:
    """Convert supported array-like objects to data accepted by h5py."""
    try:
        array = np.asarray(array)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Input array could not be converted to a \
                        NumPy array."
        ) from exc

    if array.ndim == 0:
        raise ValueError("Input array must not be scalar.")

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            f"Input array must be numeric. Got array with dtype {array.dtype!r}."
        )

    return array


def _ensure_parent_dir(filename: str) -> None:
    """Create directory if not existing"""
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _ensure_metadata(hdf5_file: h5py.File) -> None:
    """Ensure PEtab-SciML metadata group exists."""
    metadata_group = hdf5_file.require_group("metadata")

    if "pytorch_format" not in metadata_group:
        metadata_group.create_dataset("pytorch_format", data=True)
        return

    existing = bool(metadata_group["pytorch_format"][()])
    if existing is not True:
        raise ValueError(
            "Existing HDF5 file has metadata/pytorch_format=False. "
            "This writer currently writes PyTorch-format arrays."
        )


def _np_arrays_from_torch_module(
    torch_module: "torch.nn.Module",
) -> dict[str, dict[str, np.ndarray]]:
    """Extract PyTorch module parameters grouped by layer and parameter ID."""
    parameters = {}

    for name, value in torch_module.named_parameters():
        if value.numel() == 0:
            continue

        try:
            layer_id, parameter_id = name.rsplit(".", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Expected PyTorch parameter name of the form "
                f"'<layer_id>.<parameter_id>', got {name!r}."
            ) from exc

        if hasattr(value, "detach"):
            array = value.detach().numpy()
        else:
            array = _to_hdf5_data(value)
        parameters.setdefault(layer_id, {})[parameter_id] = array

    return parameters
