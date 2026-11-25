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


# TODO: Hacer que la medida tenga una función intercambiable de distancia
def density_ext(pruned: np.ndarray) -> float:
    r"""Compute the extended density of a pruned MSG (Density_ext).

    Measures how many of the possible directed edges survive in the pruned
    Maximum Similarity Graph (MSG), that is, the MSG without the edges that
    connect instances of different classes. Because the MSG is a directed
    graph, every edge is counted, and the number of possible edges is
    :math:`n(n - 1)`:

    .. math::

        Density\_ext = 1 - \frac{|E|}{n (n - 1)}

    where :math:`|E|` is the number of edges of the pruned MSG and
    :math:`n` the number of instances.

    The measure is oriented so that **lower values mean simpler data**: a
    dense pruned MSG (many surviving edges) indicates that the most similar
    instances of each point share its class.

    Parameters
    ----------
    pruned : ndarray of shape (n_samples, n_samples)
        Boolean adjacency matrix of the pruned MSG, as returned by
        :func:`prune_msg`. ``pruned[u, v]`` is ``True`` when there is a
        directed edge from instance ``u`` to its most similar instance
        ``v`` and both belong to the same class.

    Returns
    -------
    density : float
        Value of Density_ext, in the range [0, 1].

    Raises
    ------
    ValueError
        If ``pruned`` has fewer than 2 instances.

    See Also
    --------
    build_msg : Build the full MSG from a dissimilarity matrix.
    prune_msg : Remove MSG edges between instances of different classes.

    Notes
    -----
    Only the number of edges matters, not their weights, so the measure
    works with any (also non-symmetric) dissimilarity function used to
    build the MSG. It does not account for datasets where some classes are
    dense and others sparse.

    References
    ----------
    .. [1] F. J. Camacho-Urriolagoitia, Y. Villuendas-Rey,
       C. Yáñez-Márquez and O. Camacho-Nieto. "Network-based data
       complexity measures for multiclass classification of hybrid data
       and its correlation with supervised classifiers". Intelligent Data
       Analysis, SAGE Publications, 2026.
       https://doi.org/10.1177/1088467X261421339

    Examples
    --------
    Three instances with three surviving edges out of the six possible:

    >>> import numpy as np
    >>> from measures.density_ext import density_ext
    >>> pruned = np.array([[False, True, False],
    ...                    [True, False, False],
    ...                    [False, True, False]])
    >>> density_ext(pruned)
    0.5
    """
    n_edges = pruned.sum()
    n = pruned.shape[0]
    if n < 2:
        raise ValueError("pruned must have at least 2 instances.")
    return float(1 - n_edges / (n * (n - 1)))
