import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from numpy.typing import NDArray
from typing import Literal
from pathlib import Path


def plot_filter_comparison(
    original: NDArray[np.float64],
    noisy: NDArray[np.float64],
    mean_filtered: NDArray[np.uint8],
    median_filtered: NDArray[np.uint8],
    noise_type: Literal["salt_pepper", "gaussian"],
    kernel_size: int,
    output_folder: Path,
) -> None:
    """
    Create and save a comparison plot of filtering results.
    
    Parameters
    ----------
    original
        Original clean image
    noisy
        Noisy image
    mean_filtered
        Mean filtered result
    median_filtered
        Median filtered result
    noise_type
        Type of noise applied
    kernel_size
        Size of the filtering kernel
    output_folder
        Directory to save the plot
    """
    noise_name = "Salt & Pepper" if noise_type == "salt_pepper" else "Gaussian"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"{noise_name} Noise")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Original Clean Image")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(mean_filtered, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title(f"Mean Filter (kernel size={kernel_size})")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(median_filtered, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title(f"Median Filter (kernel size={kernel_size})")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    
    output_path = output_folder / f"filtering_comparison_{noise_type}_w_kernel_{kernel_size}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_individual_images(
    noisy: NDArray[np.float64],
    mean_filtered: NDArray[np.uint8],
    median_filtered: NDArray[np.uint8],
    noise_type: Literal["salt_pepper", "gaussian"],
    output_folder: Path,
) -> None:
    """
    Save individual filtered images.
    
    Parameters
    ----------
    noisy
        Noisy image with values in [0, 1]
    mean_filtered
        Mean filtered result as uint8
    median_filtered
        Median filtered result as uint8
    noise_type
        Type of noise applied
    output_folder
        Directory to save images
    """
    noisy_uint8 = (noisy * 255).astype(np.uint8)
    
    io.imsave(output_folder / f"noisy_{noise_type}.png", noisy_uint8)
    io.imsave(output_folder / f"mean_filtered_{noise_type}.png", mean_filtered)
    io.imsave(output_folder / f"median_filtered_{noise_type}.png", median_filtered)


def plot_timing_comparison(
    kernel_sizes: NDArray[np.int_],
    mean_times: NDArray[np.float64],
    median_times: NDArray[np.float64],
    noise_type: Literal["salt_pepper", "gaussian"],
    output_folder: Path,
) -> None:
    """
    Plot timing comparison for mean vs median filtering.
    
    Parameters
    ----------
    kernel_sizes
        Array of kernel sizes tested
    mean_times
        Runtime for mean filter at each kernel size
    median_times
        Runtime for median filter at each kernel size
    noise_type
        Type of noise used
    output_folder
        Directory to save the plot
    """
    noise_name = "Salt & Pepper" if noise_type == "salt_pepper" else "Gaussian"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(kernel_sizes, mean_times, 'o-', label='Mean Filter', 
            linewidth=2, markersize=6, color='#2E86AB')
    ax.plot(kernel_sizes, median_times, 's-', label='Median Filter', 
            linewidth=2, markersize=6, color='#A23B72')
    
    ax.set_xlabel('Kernel Size (N)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title(f'Filter Runtime Comparison - {noise_name} Noise\n(100 iterations)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = output_folder / f"timing_comparison_{noise_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gaussian_filtering_both_noises(
    sp_noisy: NDArray[np.float64],
    sp_filtered: NDArray[np.float64],
    gauss_noisy: NDArray[np.float64],
    gauss_filtered: NDArray[np.float64],
    sigma: float,
    kernel_size: int,
    output_folder: Path,
) -> None:
    """
    Plot Gaussian filtering results for both noise types side by side.
    
    Parameters
    ----------
    sp_noisy
        Salt & pepper noisy image
    sp_filtered
        Gaussian filtered salt & pepper image
    gauss_noisy
        Gaussian noisy image
    gauss_filtered
        Gaussian filtered gaussian noisy image
    sigma
        Standard deviation of Gaussian filter
    kernel_size
        Size of the Gaussian kernel
    output_folder
        Directory to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top row: noisy images
    axes[0, 0].imshow(sp_noisy, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Salt & Pepper Noise", fontsize=12)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(gauss_noisy, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Gaussian Noise", fontsize=12)
    axes[0, 1].axis("off")
    
    # Bottom row: filtered images
    axes[1, 0].imshow(sp_filtered, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Gaussian Filtered (S&P, K={kernel_size})", fontsize=12)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(gauss_filtered, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Gaussian Filtered (Gauss, K={kernel_size})", fontsize=12)
    axes[1, 1].axis("off")
    
    plt.suptitle(f"Gaussian Filter: σ={sigma}, kernel size K={kernel_size}", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_folder / f"gaussian_sigma{sigma}_kernel{kernel_size}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_increasing_sigma_both_noises(
    sp_noisy: NDArray[np.float64],
    sp_filtered: NDArray[np.float64],
    gauss_noisy: NDArray[np.float64],
    gauss_filtered: NDArray[np.float64],
    sigma: float,
    kernel_size: int,
    output_folder: Path,
) -> None:
    """
    Plot Gaussian filtering with increasing sigma for both noise types.
    
    Parameters
    ----------
    sp_noisy
        Salt & pepper noisy image
    sp_filtered
        Gaussian filtered salt & pepper image
    gauss_noisy
        Gaussian noisy image
    gauss_filtered
        Gaussian filtered gaussian noisy image
    sigma
        Standard deviation of Gaussian filter
    kernel_size
        Size of the Gaussian kernel
    output_folder
        Directory to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top row: noisy images
    axes[0, 0].imshow(sp_noisy, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Salt & Pepper Noise", fontsize=12)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(gauss_noisy, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Gaussian Noise", fontsize=12)
    axes[0, 1].axis("off")
    
    # Bottom row: filtered images
    axes[1, 0].imshow(sp_filtered, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Gaussian Filtered (S&P)\nK={kernel_size}, σ={sigma}", fontsize=12)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(gauss_filtered, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Gaussian Filtered (Gauss)\nK={kernel_size}, σ={sigma}", fontsize=12)
    axes[1, 1].axis("off")
    
    plt.suptitle(f"Gaussian Filter: σ={sigma}, kernel size={kernel_size}", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_folder / f"increasing_sigma{sigma}_kernel{kernel_size}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
