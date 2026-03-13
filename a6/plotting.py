from typing import Any, Callable
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def plot_kmeans_segmentation(
    image: NDArray[Any], 
    segmentation: NDArray[Any],
    save_path: Path,
    title: str = "PCA Filter Bank Segmentation (K=3)",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(segmentation, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title(title)
    axes[1].axis("off")

    # Add legend for the 3 clusters
    cmap = plt.get_cmap("tab10")
    legend_patches = [
        mpatches.Patch(color=cmap(i), label=f"Cluster {i+1}")
        for i in range(3)
    ]
    axes[1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.8
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_translation(
    image: NDArray[np.uint8],
    x_trans: float | int,
    y_trans: float | int,
    translation_function: Callable,
    save_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(1, 2)
    # Plot square and translated square
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title(f"Original Image of size {image.shape[0]}x{image.shape[1]}")
    ax[0].axis("off")
    ax[1].imshow(translation_function(image, x_trans, y_trans), cmap="gray")
    ax[1].set_title(title)
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_grid(
    images: NDArray[Any] | list[NDArray[Any]], save_path: Path, title: str
) -> None:
    # Vars
    assert np.sqrt(len(images)) % 1 == 0, "Number of images must be a perfect square"
    sqrt = int(np.sqrt(len(images)))
    subplots = (sqrt, sqrt)
    figsize = (sqrt + 1, sqrt + 1)

    fig, axes = plt.subplots(*subplots, figsize=figsize)
    fig.suptitle(
        title,
        fontsize=16
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")

    # plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, hspace=0.05, wspace=0.05)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
