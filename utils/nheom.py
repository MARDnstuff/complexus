"""NHEOM definition and auxiliar functions."""

import math
import warnings
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd


# TODO: Move validations to a higher level
def compute_ranges(
    data: pd.DataFrame,
    is_categorical: Sequence[bool],
) -> list[float | None]:
    """Compute the range of each numeric column."""
    if len(is_categorical) != data.shape[1]:
        raise ValueError(
            f"is_categorical has {len(is_categorical)} entries but data has "
            f"{data.shape[1]} columns."
        )
    rng: list[float | None] = []
    for i, cat in enumerate(is_categorical):
        if cat:
            rng.append(None)
            continue

        col = data.iloc[:, i]

        if col.isna().all():
            warnings.warn(
                f"Column '{data.columns[i]}' is entirely missing; "
                "it contributes 1 to NHEOM for every pair.",
                stacklevel=2,
            )
            rng.append(None)
            continue

        value_range = float(col.max() - col.min())
        if value_range == 0:
            warnings.warn(
                f"Column '{data.columns[i]}' is constant; "
                "it contributes 0 to NHEOM.",
                stacklevel=2,
            )
        rng.append(value_range)
    return rng


def rd_diss(a: float, b: float, value_range: float) -> float:
    """Dissimilarity between two values of a numeric attribute."""
    # Negative range is invalid
    if value_range < 0:
        raise ValueError(
            f"value_range must be non-negative, got {value_range}."
        )

    # Constant attribute: all values are equal, so the difference is 0
    if value_range == 0:
        if a != b:
            raise ValueError(
                f"value_range is 0 but a={a} and b={b} differ; "
                "the range does not match these values."
            )
        return 0.0
    return float(np.abs(a - b) / value_range)


def overlap(a: Hashable, b: Hashable) -> float:
    """Dissimilarity between two values of a categorical attribute."""
    return float(a != b)


def diss(
    a: Hashable | float,
    b: Hashable | float,
    is_categorical: bool,
    value_range: float | None = None,
) -> float:
    """Distance between two values 'a' and 'y' of a given attribute."""
    if pd.isna(a) or pd.isna(b):
        return 1.0
    if is_categorical:
        return overlap(a, b)
    if value_range is None:
        raise ValueError("value_range is required for numeric attributes.")
    return rd_diss(a, b, value_range)


def nheom(
    x: Sequence[Hashable | float],
    y: Sequence[Hashable | float],
    is_categorical: Sequence[bool],
    value_ranges: Sequence[float | None],
) -> float:
    """
    Compute the NHEOM dissimilarity between two instances.

    NHEOM
    """
    m = len(x)
    if m == 0:
        raise ValueError("x and y must have at least one attribute.")

    acc = 0.0
    # strict=True raises ValueError if lengths differ
    for a, b, cat, rng in zip(x, y, is_categorical, value_ranges, strict=True):
        acc += diss(a, b, cat, rng) ** 2
    return math.sqrt(acc) / math.sqrt(m)


def nheom_matrix(
    data: pd.DataFrame,
    is_categorical: Sequence[bool],
) -> np.ndarray:
    """Compute the NHEOM dissimilarity matrix between all instances."""
    value_ranges = compute_ranges(data, is_categorical)
    rows = [data.iloc[u] for u in range(len(data))]
    n = len(rows)

    diss_matrix = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            diss_matrix[u, v] = nheom(
                rows[u], rows[v], is_categorical, value_ranges
            )
    return diss_matrix
