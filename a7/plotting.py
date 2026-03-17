import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.color import rgb2gray
import numpy as np
from numpy.typing import NDArray
from typing import Any
from pathlib import Path


def plot_open_close(original, closed, opened, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(closed, cmap="gray")
    axes[1].set_title("Closed Image")
    axes[1].axis("off")
    axes[2].imshow(opened, cmap="gray")
    axes[2].set_title("Opened Image")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_overlay_labels(image, labels, save_path):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    # For each unique label (skip 0 if it's background)
    for label_id in np.unique(labels):
          if label_id == 0:
              continue

          # Find centroid of the label region
          mask = labels == label_id
          cy, cx = ndimage.center_of_mass(mask)

          ax.text(cx, cy, str(label_id),
                  color="red", fontsize=10, fontweight="bold",
                  ha="center", va="center",
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_w_and_wo_color(image: NDArray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title(f"{title} (color)", fontsize=20)
    ax[0].axis("off")
    ax[1].imshow(rgb2gray(image), cmap="gray")
    ax[1].set_title(f"{title} (grayscale)", fontsize=20)
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_histogram(
    img: NDArray[np.uint8],
    img_name: str,
    save_path: Path,
    threshold: float | None = None,
) -> None:
    assert img.ndim == 2, "Input image must be grayscale"
    plt.hist(img.ravel(), bins=256, range=(0, 255), label="Histogram of Grayscale Image")
    # Plot a vertical dotted line at x=100
    if threshold is not None:
        plt.axvline(x=threshold, color="r", linestyle="--", label=f"Threshold={threshold:.2f}")
    title = f"Histogram of Grayscale Image {img_name}"
    title += "with Otsu Threshold" if threshold is not None else ""
    plt.title(f"Histogram of Grayscale Image '{img_name}'", fontsize=14)
    plt.legend()
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    plt.grid()
    plt.savefig(save_path)
    plt.close()
