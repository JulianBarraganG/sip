import numpy as np
from numpy.typing import NDArray
from skimage import io
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

from const import IMAGES_FOLDER, OUTPUT_FOLDER


def normalized_laplacian_stack(
    image: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the scale-normalised Laplacian stack using gaussian_filter with
    derivative orders.  H(x,y,τ) = τ² · (Ixx + Iyy),  γ=1, m+n=2.

    Parameters
    ----------
    image  
        2-D float grayscale image
    scales 
        1-D array of τ values in pixels

    Returns
    -------
    stack  
        (len(scales), H, W) scale-space of the image
    """
    stack = []
    for tau in scales:
        Ixx = gaussian_filter(image, sigma=tau, order=(0, 2))
        Iyy = gaussian_filter(image, sigma=tau, order=(2, 0))
        H = tau**2 * (Ixx + Iyy)
        stack.append(H)
    return np.array(stack)


def detect_blobs(
    stack: NDArray[np.float64],
    scales: NDArray[np.float64],
    n_blobs: int = 150,
    min_distance: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Find the n_blobs extrema with the largest |H| in the scale-space stack.

    Runs peak_local_max on both the stack and its negation to find maxima and
    minima separately, then merges and keeps the top n_blobs by |H|.

    Parameters
    ----------
    stack
        (S, H, W) Laplacian stack
    scales
        (S,) τ values corresponding to stack axis 0
    n_blobs
        total number of detections to return
    min_distance
        minimum pixel distance between detections in (scale, y, x)

    Returns
    -------
    coords
        (n_blobs, 3) array of (scale_idx, row, col)
    labels
        (n_blobs,)  array of strings, "max" or "min"
    taus
        (n_blobs,)  array of τ values for each detection
    """
    # Generous initial pool so we can rank by |H| afterwards
    pool = n_blobs * 5

    max_coords = peak_local_max(
        stack, num_peaks=pool, min_distance=min_distance, exclude_border=False
    )
    min_coords = peak_local_max(
        -stack, num_peaks=pool, min_distance=min_distance, exclude_border=False
    )
    
    # Max (min) coords will be shape (N_max, 3) with (scale_idx, row, col)
    max_vals = np.abs(stack[max_coords[:, 0], max_coords[:, 1], max_coords[:, 2]])
    min_vals = np.abs(stack[min_coords[:, 0], min_coords[:, 1], min_coords[:, 2]])

    all_coords = np.vstack([max_coords, min_coords])
    all_vals   = np.concatenate([max_vals, min_vals])
    all_labels = np.array(["max"] * len(max_coords) + ["min"] * len(min_coords))

    # Keep top n_blobs by |H|
    top_idx    = np.argsort(all_vals)[::-1][:n_blobs]
    coords     = all_coords[top_idx]
    labels     = all_labels[top_idx]
    taus       = scales[coords[:, 0]]

    return coords, labels, taus



if __name__ == "__main__":
    from plotting import plot_blobs
    import matplotlib
    matplotlib.use("Agg")

    OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)

    # --- Load image ---------------------------------------------------------
    img_path = IMAGES_FOLDER / "sunflower.tiff"
    image = io.imread(img_path).astype(np.float64)

    print(f"Image shape: {image.shape},  dtype: {image.dtype}")

    # --- Scale-space --------------------------------------------------------
    SCALES = np.linspace(2, 30.0, 20)
    print(f"Building Laplacian stack for {len(SCALES)} scales: "
          f"τ ∈ [{SCALES[0]:.1f}, {SCALES[-1]:.1f}] …")

    stack = normalized_laplacian_stack(image, SCALES)
    print(f"Stack shape: {stack.shape},  |H| max: {np.abs(stack).max():.4f}")

    # --- Detect blobs -------------------------------------------------------
    print("Detecting blobs …")
    coords, labels, taus = detect_blobs(stack, SCALES, n_blobs=150)
    print(f"  Maxima: {np.sum(labels=='max')}   Minima: {np.sum(labels=='min')}")
    print(f"  τ range of detections: [{taus.min():.1f}, {taus.max():.1f}]")

    # --- Plot ---------------------------------------------------------------
    print("Plotting …")
    plot_blobs(
        image=image,
        coords=coords,
        labels=labels,
        taus=taus,
        output_path=OUTPUT_FOLDER / "blob_detection.png",
    )

    print("Done.")
