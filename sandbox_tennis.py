"""Sandbox."""

import numpy as np
import pandas as pd

from measures.density_ext import build_msg, density_ext, prune_msg
from utils.nheom import nheom_matrix

if __name__ == "__main__":
    df = pd.read_csv("dataset/tennis_camacho2023.csv")
    X = df.drop(columns="Play")
    y = df["Play"]
    is_categorical = [True, True, True, False]

    diss_matrix = nheom_matrix(X, is_categorical)
    print(diss_matrix)

    diag = np.diag(diss_matrix)
    print(diag)
    np.flatnonzero(diag)  # índices de las 3 instancias con faltantes
    diag[diag > 0]

    adj = build_msg(diss_matrix)
    pruned = prune_msg(adj, y)
    print("Tennis density_ext: ", density_ext(pruned=pruned))

    print("MSG completo:", adj.sum())
    print("MSG podado:  ", pruned.sum())
    print("Grado de salida (completo):", np.bincount(adj.sum(axis=1)))
    print("Clases:", pd.Series(y).value_counts().to_dict())
    print("Etiquetas exactas:", [repr(c) for c in np.unique(y)])
    print("Filas duplicadas:", X.duplicated(keep=False).sum())

    y_arr = np.asarray(y)
    same = y_arr[:, None] == y_arr[None, :]
    cross = adj & ~same

    I1 = ~(adj & same).any(axis=1)  # NetI: ninguna arista a su clase
    I2 = cross.any(axis=1)  # NetPI: al menos una a otra clase
    I3 = cross.sum(axis=1) >= (adj & same).sum(
        axis=1
    )  # NetC: enemigos >= amigos
    I4 = cross.any(axis=0)  # NetB: alguien de otra clase lo eligió

    for name, ind in [("NetI", I1), ("NetPI", I2), ("NetC", I3), ("NetB", I4)]:
        per_class = [ind[y_arr == c].mean() for c in np.unique(y_arr)]
        print(
            f"{name}: {ind.mean():.2f}  min {min(per_class):.2f}  "
            f"avg {np.mean(per_class):.2f}  max {max(per_class):.2f}"
        )

    pred = adj.T  # pred[u, v]: v eligió a u
    neigh = adj | pred  # sucesores ∪ predecesores
    enemies = (neigh & ~same).sum(axis=1)
    friends = (neigh & same).sum(axis=1)
    I3b = enemies >= friends

    per_class = [I3b[y_arr == c].mean() for c in np.unique(y_arr)]
    print(
        f"NetC (succ+pred): {I3b.mean():.2f}  min {min(per_class):.2f}  "
        f"avg {np.mean(per_class):.2f}  max {max(per_class):.2f}"
    )

    enemies2 = (adj & ~same).sum(axis=1) + (pred & ~same).sum(axis=1)
    friends2 = (adj & same).sum(axis=1) + (pred & same).sum(axis=1)
    I3c = enemies2 >= friends2
    per_class = [I3c[y_arr == c].mean() for c in np.unique(y_arr)]
    print(
        f"NetC (succ+pred, doble): {I3c.mean():.2f}  min {min(per_class):.2f}  "
        f"avg {np.mean(per_class):.2f}  max {max(per_class):.2f}"
    )
