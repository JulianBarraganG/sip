import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path

from threshold import threshold_image

def plot_thresholding(image: NDArray, save_path: Path, threshold: int) -> None:
    """
    Plot the three images side by side [Original, Gray, Binary]
    """
    binary_mask = threshold_image(image, threshold)
    # plot original image, grayscale image and binary mask
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(np.mean(image, axis=2).astype(np.uint8), cmap="gray")
    axes[1].set_title("Grayscale Image")
    axes[1].axis("off")
    axes[2].imshow(binary_mask, cmap="gray")
    axes[2].set_title("Binary Segmentation Image")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
