from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, RootModel

from mkstd import Hdf5Standard
from mkstd.types.array import get_array_type


__all__ = [
    "ConditionSpecificSingleInputData",
    "SingleInputData",
    "Metadata",
    "ArrayData",
    "ArrayDataStandard",
]


DATA = "data"
CONDITION_IDS = "conditionIds"


Array = get_array_type()


class ConditionSpecificSingleInputData(BaseModel):
    """Condition-specific input data for a single input."""

    data: Array
    """The data."""

    conditionIds: list[str] | None = Field(default=None)
    """The dataset is used with these conditions.

    The default (`None`) indicates all conditions.
    """


SingleInputData = RootModel[list[ConditionSpecificSingleInputData]]
"""All input data for a single input."""


class Metadata(BaseModel):
    """Metadata for array(s)."""

    perm: Literal["row", "column"]
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

    inputs: dict[str, SingleInputData] = {}
    """Input data arrays.

    Keys are input IDs, values are the input data arrays and their applicable
    conditions.
    """

    parameters: dict[str, dict[str, Array]] = {}
    """Parameter value arrays.

    Outer dict keys are layer IDs. Inner dict keys are layer-specific parameter
    IDs, and inner dict values are the parameter value arrays.
    """


ArrayDataStandard = Hdf5Standard(model=ArrayData)


if __name__ == "__main__":
    from pathlib import Path


    ArrayDataStandard.save_schema(
        Path(__file__).resolve().parents[4]
        / "docs" / "src" / "assets" / "array_data_schema.json"
    )
