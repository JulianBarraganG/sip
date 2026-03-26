from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import polars as pl

def plot_wings(
    coords: pl.DataFrame, aligned: NDArray[Any], save_path: Path, num_wings: int = 10,
) -> None:
    # Make side-by-side plots of original and aligned wings
    fix, ax = plt.subplots(1, 2, figsize=(18, 6))
    for i in range(num_wings):
        color_i = plt.get_cmap("Set1")(i)
        row = coords.row(i) # i-th row
        ax[0].plot(row[0::2], row[1::2], marker=".", label=f"Wing {i+1}", color=color_i)
        ax[0].set_title("Original Wings")
        ax[1].plot(aligned[i, 0, :], aligned[i, 1, :], marker=".", label=f"Aligned Wing {i+1}", color=color_i)
        ax[1].set_title("Aligned Wings")


    plt.title(f"Wing Landmarks of {num_wings} wings")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
