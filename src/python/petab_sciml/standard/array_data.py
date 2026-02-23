from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, RootModel, field_validator, model_serializer

from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type


__all__ = [
    "Metadata",
    "ArrayData",
    "ArrayDataStandard",
    "METADATA",
    "DATA",
    "CONDITION_IDS",
    "PERM",
    "INPUTS",
    "PARAMETERS",
    "ROW",
    "COLUMN",
    "ALL_CONDITION_IDS",
]


METADATA = "metadata"
DATA = "data"
CONDITION_IDS = "conditionIds"
INPUTS = "inputs"
PARAMETERS = "parameters"
PERM = "perm"
ROW = "row"
COLUMN = "column"
ALL_CONDITION_IDS = "0"


Array = get_array_type()


class Metadata(BaseModel):
    """Metadata for array(s)."""

    perm: Literal[ROW, COLUMN]
    """The order of the storage of arrays.

    i.e., row-major or column-major arrays.
    """

    pytorch_dimension_order: bool
    """The order of the dimensions of arrays.

    i.e., a weight matrix could be stored as `in_features x out_features` in
    one framework, and `out_features x in_features` in another framework.

    `True` indicates that the order of the dimensions of arrays matches the
    order used by PyTorch. `False` indicates an unspecified ordering.
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

    @field_validator(INPUTS, mode='after')
    @classmethod
    def validate_condition_ids(cls, inputs) -> dict[str, SingleInputData]:
        for input_id, input_data in inputs.items():
            if not input_data:
                raise ValueError(
                    f"No input data supplied for input `{input_id}`."
                )

            for condition_ids_str, array in input_data.items():
                n_arrays = len(input_data)
                if (
                    (condition_ids_str == ALL_CONDITION_IDS) and
                    (n_arrays != 1)
                ):
                        raise ValueError(
                            "The condition IDs list is "
                            f"`{ALL_CONDITION_IDS}`, which indicates that the "
                            "array will be applied to all conditions. In this "
                            "case, exactly one array must be specified. "
                            f"However, {n_arrays} arrays were specified for "
                            f"input `{input_id}`."
                        )
        return inputs


ArrayDataStandard = Hdf5Standard(model=ArrayData)


if __name__ == "__main__":
    from pathlib import Path


    ArrayDataStandard.save_schema(
        Path(__file__).resolve().parents[4]
        / "doc" / "standard" / "array_data_schema.json"
    )
