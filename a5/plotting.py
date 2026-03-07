from const import OUTPUT_FOLDER
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def plot_tau_range(
    blob: NDArray[np.float64],
    taus: list[float],
    grid_half_width: int,
    scale_space_images: list[NDArray[np.float64]],
    sigma: float = 1.0,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(taus) + 1,
        figsize=(3.2 * (len(taus) + 1), 3.8),
    )

    axes[0].imshow(blob, cmap="gray", extent=[-grid_half_width, grid_half_width] * 2)
    axes[0].set_title(f"B(x,y)\nσ = {sigma}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    for ax, tau, img in zip(axes[1:], taus, scale_space_images):
        ax.imshow(img, cmap="gray", extent=[-grid_half_width, grid_half_width] * 2)
        ax.set_title(f"I(x,y,τ={tau})")
        ax.set_xlabel("x")
        ax.axis("on")

    fig.suptitle(
        "Scale-space of blob B(x,y) = G(x,y,σ=1): I(x,y,τ) = B * G(x,y,τ)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "4_1_scale_space.png", bbox_inches="tight", dpi=150)
    plt.close()
