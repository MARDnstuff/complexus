"""Sandbox."""

import numpy as np
import pandas as pd

from utils.nheom import nheom_matrix


def get_dataset(file_path: str, columns: list[str]) -> pd.DataFrame:
    """Get Dataset from file_path."""
    try:
        # Assuming data is in a structured format like CSV or similar
        df = pd.read_csv(
            file_path,
            comment="@",
            header=None,
            names=columns,
            na_values=["?", "<null>"],
            skipinitialspace=True,
            dtype=str,
        )
        # Process the DataFrame
        return df

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_path = "dataset/post-operative.dat"
    columns = [
        "L-CORE",
        "L-SURF",
        "L-O2",
        "L-BP",
        "SURF-STBL",
        "CORE-STBL",
        "BP-STBL",
        "COMFORT",
        "Decision",
    ]

    df = get_dataset(file_path=file_path, columns=columns)

    X = df.drop(columns="Decision")
    y = df["Decision"]

    is_categorical = [
        True,  # L-CORE     {mid, high, low}
        True,  # L-SURF     {low, high, mid}
        True,  # L-O2       {excellent, good}
        True,  # L-BP       {mid, high, low}
        True,  # SURF-STBL  {stable, unstable}
        True,  # CORE-STBL  {stable, unstable, mod-stable}
        True,  # BP-STBL    {stable, mod-stable, unstable}
        True,  # COMFORT    {15, 10, 05, 07}
    ]

    res = nheom_matrix(X, is_categorical)
    print(res)

    diag = np.diag(res)
    print(diag)
    np.flatnonzero(diag)  # índices de las 3 instancias con faltantes
    diag[diag > 0]
