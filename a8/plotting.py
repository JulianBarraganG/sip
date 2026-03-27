import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Any


def plot_square_vs_transformed(
    sq_img: NDArray[Any], transformed_img: NDArray[Any], save_path: Path,
) -> None:
    # plot side by side the original and transformed images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(sq_img, cmap="gray")
    ax[0].set_title("Original White Square Image")
    ax[0].axis("off")
    ax[1].imshow(transformed_img, cmap="gray")
    ax[1].set_title(r"Transformation: $T_tT_cRST_c^{-1}$ on White Square Image")
    ax[1].axis("off")

    fig.suptitle("Original vs Transformed White Square Image", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_wings(
    coords: pl.DataFrame, aligned: NDArray[Any], save_path: Path, num_wings: int = 10,
) -> None:
    # Make side-by-side plots of original and aligned wings
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for i in range(num_wings):
        color_i = plt.get_cmap("Set1")(i)
        row = coords.row(i) # i-th row
        ax[0].plot(row[0::2], row[1::2], marker=".", label=f"Wing {i+1}", color=color_i)
        ax[0].set_title("Original Wings")
        ax[0].legend()
        ax[1].plot(aligned[i, 0, :], aligned[i, 1, :], marker=".", label=f"Aligned Wing {i+1}", color=color_i)
        ax[1].set_title("Aligned Wings")
        ax[1].legend()


    fig.suptitle(f"Wing Landmarks of {num_wings} wings", fontsize=22)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    #plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_original_vs_seg(
    image: NDArray[np.uint16],
    seg_mask: NDArray[np.float32],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    plt.imshow(seg_mask, cmap="nipy_spectral", vmin=0, vmax=134) # 135 classes
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_true_vs_pred_seg(
    true_seg: NDArray[np.float32],
    pred_seg: NDArray[np.float32],
    save_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Segmentation")
    plt.imshow(true_seg, cmap="nipy_spectral")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Model Predicted Segmentation")
    plt.imshow(pred_seg, cmap="nipy_spectral")
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()
