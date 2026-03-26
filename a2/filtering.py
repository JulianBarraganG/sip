import numpy as np
from skimage import io
from skimage.util import random_noise
from skimage.filters.rank import median, mean
from skimage.morphology import square
from scipy.ndimage import gaussian_filter
from time import time

from numpy.typing import NDArray
from typing import Literal, Any
from plotting import (
    plot_filter_comparison,
    plot_timing_comparison,
    plot_gaussian_filtering_both_noises,
    plot_increasing_sigma_both_noises,
)


def add_noise(
    image: NDArray[np.float64],
    noise_type: Literal["salt_pepper", "gaussian"],
    sp_amount: float = 0.05,
    var: float = 0.01,
) -> NDArray[np.float64]:
    """
    Add noise to an image.
    
    Parameters
    ----------
    image
        Clean image with values in [0, 1]
    noise_type
        Type of noise to add
    sp_amount
        Amount of salt & pepper noise (ignored for gaussian), range is [0, 1]
    var
        Variance of gaussian noise (ignored for salt & pepper)
        
    Returns
    -------
    NDArray[np.float64]
        Noisy image with values in [0, 1]
    """
    assert np.max(image) <= 1., (
        f"Input image expected range [0, 1], found max: {np.max(image)}"
    )
    if noise_type == "salt_pepper":
        return random_noise(image, mode="s&p", amount=sp_amount)
    else:  # gaussian
        return random_noise(image, mode="gaussian", var=var)


def apply_filters(
    noisy_image: NDArray[np.float64],
    footprint: NDArray[np.bool_],
) -> tuple[NDArray[Any | np.uint8], NDArray[Any | np.uint8]]:
    """
    Apply mean and median filters to a noisy image.
    
    Parameters
    ----------
    noisy_image
        Noisy image with values in [0, 1]
    footprint
        Binary mask defining the neighborhood structure
        
    Returns
    -------
    tuple[NDArray[np.uint8], NDArray[np.uint8]]
        (mean_filtered, median_filtered) images as uint8
    """
    assert np.max(noisy_image) <= 1., (
        f"Noisy image expected range [0, 1], found max: {np.max(noisy_image)}"
    )
    # Convert to uint8 for rank filters
    noisy_uint8 = (noisy_image * 255).astype(np.uint8)
    
    mean_filtered = mean(noisy_uint8, footprint)
    median_filtered = median(noisy_uint8, footprint)
    
    return mean_filtered, median_filtered

def run_num_rounds(
    noisy_image: NDArray[np.float64],
    footprint: NDArray[np.bool_],
    filter_type: Literal["mean", "median"],
    num_rounds: int = 100,
    verbose: bool = False,
) -> float:
    """
    Apply a specific filter num_rounds times and return average time.
    
    Parameters
    ----------
    noisy_image
        Noisy image with values in [0, 1]
    footprint
        Binary mask defining the neighborhood structure
    filter_type
        Which filter to benchmark
    num_rounds
        Number of iterations to run
    verbose
        Whether to print timing results
        
    Returns
    -------
    float
        Total time in seconds for all rounds
    """
    assert np.max(noisy_image) <= 1., (
        f"Noisy image expected range [0, 1], found max: {np.max(noisy_image)}"
    )
    noisy_uint8 = (noisy_image * 255).astype(np.uint8)
    
    start = time()
    for _ in range(num_rounds):
        if filter_type == "mean":
            mean(noisy_uint8, footprint)
        else:  # median
            median(noisy_uint8, footprint)
    total_time = time() - start
    if verbose:
        print(
            f"{filter_type.capitalize()} filter with kernel size"
            f"{footprint.shape[0]}x{footprint.shape[1]}:"
            f" {total_time:.4f} seconds for {num_rounds} rounds"
        )
    
    return total_time

if __name__ == "__main__":
    from tqdm import tqdm
    from const import DATA_FOLDER, OUTPUT_FOLDER
    # Global variables
    N = range(1, 26)  # Kernel size
    RUN_COMPARISON = False
    RUN_TIMINGS = False
    RUN_FIXED_STD = True
    RUN_INC_STD = False
    # If folder doesn't exist, make it
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Load the image
    image_folder = DATA_FOLDER / "images"
    image_path = image_folder / "eight.tif"
    assert image_path.exists(), f"Image not found: {image_path}"
    image = io.imread(image_path)
    assert len(image.shape) != 3, (
        "Current implementation assumes grayscale i.e. 1 channel"
        f" not {len(image.shape)}"
    )
    
    # Normalize to [0, 1] range
    image = image.astype(np.float64) / 255.0

    # Assignment 3.3, increasing sigma with N = 3*sigma
    if RUN_INC_STD:
        sigmas = [1, 3, 5, 7]
        save_folder = OUTPUT_FOLDER / "gaussian_increasing_sigma"
        save_folder.mkdir(parents=False, exist_ok=True)
        
        # Generate noisy images once
        sp_noisy_image = add_noise(image, "salt_pepper")
        gauss_noisy_image = add_noise(image, "gaussian")
        
        for sigma in tqdm(sigmas, desc="Increasing sigma with N=3σ"):
            # Kernel size N = 3*sigma, radius = (N-1)/2 = (3*sigma - 1)/2
            radius = int((3*sigma - 1) / 2)
            kernel_size = 2*radius + 1 # see docs gaussian_filter
            
            # Filter both noisy images
            sp_filtered = gaussian_filter(sp_noisy_image, sigma=sigma, radius=radius)
            gauss_filtered = gaussian_filter(gauss_noisy_image, sigma=sigma, radius=radius)
            
            # Plot side by side
            plot_increasing_sigma_both_noises(
                sp_noisy=sp_noisy_image,
                sp_filtered=sp_filtered,
                gauss_noisy=gauss_noisy_image,
                gauss_filtered=gauss_filtered,
                sigma=sigma,
                kernel_size=kernel_size,
                output_folder=save_folder,
            )

    # Assignment 3.2, fixed std=5 and increasing kernel sizes
    if RUN_FIXED_STD:
        std = 5
        radii = range(1, 13)  # Corresponding to kernel sizes 3, 5, ..., 25
        save_folder = OUTPUT_FOLDER / "gaussian_fixed_std"
        # Make dir if it doesn't exist, but don't overwrite existing results
        save_folder.mkdir(parents=False, exist_ok=True)
        
        # Generate noisy images once
        sp_noisy_image = add_noise(image, "salt_pepper")
        gauss_noisy_image = add_noise(image, "gaussian")
        
        for radius in tqdm(radii, desc=f"Fixed std={std}, varying kernel size"):
            kernel_size = 2*radius + 1 # see docs gaussian_filter
            
            # Filter both noisy images
            sp_filtered = gaussian_filter(sp_noisy_image, sigma=std, radius=radius)
            gauss_filtered = gaussian_filter(gauss_noisy_image, sigma=std, radius=radius)
            
            # Plot side by side
            plot_gaussian_filtering_both_noises(
                sp_noisy=sp_noisy_image,
                sp_filtered=sp_filtered,
                gauss_noisy=gauss_noisy_image,
                gauss_filtered=gauss_filtered,
                sigma=std,
                kernel_size=kernel_size,
                output_folder=save_folder,
            )


    # Assignment 3.1 part 1
    if RUN_COMPARISON:
        save_folder = OUTPUT_FOLDER / "filtering"
        # Run for all kernel sizes and save comparison plot
        for k_size in tqdm(N, desc="Initial Comparison Plots"):
            for noise_type in ["salt_pepper", "gaussian"]:
                noisy_image = add_noise(image, noise_type)
                mean_filtered, median_filtered = apply_filters(
                    noisy_image,
                    footprint=square(k_size),
                )
                plot_filter_comparison(
                    original=image,
                    noisy=noisy_image,
                    mean_filtered=mean_filtered,
                    median_filtered=median_filtered,
                    noise_type=noise_type,
                    kernel_size=k_size,
                    output_folder=save_folder,
                )

    # Assignment 3.1 part 2
    if RUN_TIMINGS:
        # Execute for each kernel and each filter 100 times and save results
        sp_mean_times = np.zeros(len(N))
        sp_median_times = np.zeros(len(N))
        gauss_mean_times = np.zeros(len(N))
        gauss_median_times = np.zeros(len(N))
        sp_noisy_image = add_noise(image, "salt_pepper")
        gauss_noisy_image = add_noise(image, "gaussian")
        tqdm_desc = "For each kernel size K time both filters on both noise types"
        for i, k_size in tqdm(enumerate(N), desc=tqdm_desc):
            kernel = square(k_size)
            sp_mean_times[i] = run_num_rounds(
                sp_noisy_image, footprint=kernel, filter_type="mean",num_rounds=100,
            )
            sp_median_times[i] = run_num_rounds(
                sp_noisy_image, footprint=kernel, filter_type="median", num_rounds=100,
            )
            gauss_mean_times[i] = run_num_rounds(
                gauss_noisy_image, footprint=kernel, filter_type="mean", num_rounds=100,
            )
            gauss_median_times[i] = run_num_rounds(
                gauss_noisy_image, footprint=kernel, filter_type="median", num_rounds=100,
            )

            # Plot timing comparisons
            kernel_sizes = np.array(list(N))
            plot_timing_comparison(
                kernel_sizes, sp_mean_times, sp_median_times, "salt_pepper", OUTPUT_FOLDER,
            )
            plot_timing_comparison(
                kernel_sizes, gauss_mean_times, gauss_median_times, "gaussian", OUTPUT_FOLDER,
            )

    if not(RUN_COMPARISON or RUN_TIMINGS or RUN_FIXED_STD or RUN_INC_STD):
        print("WARNING: No boolean 'RUN_XYZ' was set to true, script did nothing.")
