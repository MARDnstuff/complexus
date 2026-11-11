"""Extended dentisty of the network definition."""

import numpy as np
import numpy.typing as npt


def build_msg(dissimilarity: np.ndarray) -> np.ndarray:
    """Build the full MSG from a dissimilarity matrix."""
    if not (
        dissimilarity.ndim == 2
        and dissimilarity.shape[0] == dissimilarity.shape[1]
    ):
        raise ValueError(
            "dissimilarity must be a square 2-D array, "
            f"got shape {dissimilarity.shape}."
        )

    if dissimilarity.shape[0] < 2:
        raise ValueError("dissimilarity must have at least 2 instances.")

    diss_copy = dissimilarity.astype(float, copy=True)
    # fill diagonal with inf to exclude it
    np.fill_diagonal(diss_copy, np.inf)
    # Get the min value of each row
    min_val_row = diss_copy.min(axis=1, keepdims=True)
    adj = np.isclose(diss_copy, min_val_row, rtol=0, atol=1e-12)
    return adj


def prune_msg(adjacency: np.ndarray, y: npt.ArrayLike) -> np.ndarray:
    """Remove MSG edges between instances of different classes."""
    y = np.asarray(y)
    if y.shape[0] != adjacency.shape[0]:
        raise ValueError(
            f"y has {y.shape[0]} labels but the MSG has "
            f"{adjacency.shape[0]} instances."
        )
    same_class = y[:, None] == y[None, :]
    return adjacency & same_class


def density_ext(pruned: np.ndarray) -> float:
    """Compute the extended density of a pruned MSG (Density_ext)."""
    n_edges = pruned.sum()
    n = pruned.shape[0]
    if n < 2:
        raise ValueError("pruned must have at least 2 instances.")
    return float(1 - n_edges / (n * (n - 1)))
