from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, RootModel, field_validator, model_serializer

from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type


__all__ = [
    "ConditionSpecificSingleInputData",
    "SingleInputData",
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
]


METADATA = "metadata"
DATA = "data"
CONDITION_IDS = "conditionIds"
INPUTS = "inputs"
PARAMETERS = "parameters"
PERM = "perm"
ROW = "row"
COLUMN = "column"


Array = get_array_type()


class Metadata(BaseModel):
    """Metadata for array(s)."""

    perm: Literal[ROW, COLUMN]
    """The order of the dimensions of arrays.

    i.e., row-major or column-major arrays.
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
            root_data = input_data.root
            if not root_data:
                raise ValueError(
                    f"No input data supplied for input `{input_id}`."
                )

            if len(root_data) == 1:
                if list(root_data.values())[0].conditionIds:
                    raise ValueError(
                        "Do not specify condition IDs if supplying only one "
                        "array for an input. When only one array is supplied, "
                        "it is applied to all conditions. "
                        f"Input: `{input_id}`."
                    )
            else:
                for array_dict in root_data.values():
                    if not array_dict.conditionIds:
                        raise ValueError(
                            "Condition IDs must be specified for each array, "
                            "when multiple arrays are supplied for a single "
                            f"input. Input: `{input_id}`."
                        )
                if list(root_data.keys()) != list(map(str, range(len(root_data)))):
                    raise ValueError(
                        "The keys of the condition-specific array data for a "
                        "single input must be ascending from 0. "
                        f"Input: `{input_id}`."
                    )
        return inputs


ArrayDataStandard = Hdf5Standard(model=ArrayData)


if __name__ == "__main__":
    from pathlib import Path


    ArrayDataStandard.save_schema(
        Path(__file__).resolve().parents[4]
        / "doc" / "standard" / "array_data_schema.json"
    )
