from typing import Callable
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt


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
