from typing import Any
from numpy.typing import NDArray

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA


def pca_bank(
    image: NDArray[Any],
    patch_size: tuple[int, int],
    num_filters: int,
    verbose: bool = True,
) -> NDArray[Any]:
    assert image.ndim == 2, "Image must be 2D grayscale"
    image_patches = extract_patches_2d(image, patch_size)  # (n_patches, 8, 8)
    n_patches = image_patches.shape[0]
    input_patches = image_patches.reshape(n_patches, -1)   # (n_patches, 64)

    # Train PCA, keep all components
    pca = PCA()
    pca.fit(input_patches)

    # Top num_filters principal components reshaped to 8x8
    if verbose:
        expl_var = pca.explained_variance_
        print(
            f"Explained variance in percent list:"
                f"\n{expl_var[:num_filters] / expl_var.sum() * 100}"
        )
        print(
            f"Cummulative var explained for {num_filters} filters: "
            f"{expl_var[:num_filters].sum() / expl_var.sum() * 100:.4f}%"
        )
    return pca.components_[:num_filters].reshape(num_filters, *patch_size)


if __name__ == "__main__":
    from const import IMAGES_FOLDER, OUTPUT_FOLDER
    from skimage.io import imread
    from plotting import plot_grid, plot_kmeans_segmentation
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from scipy.signal import fftconvolve
    import numpy as np

    PATCH_SIZE = (8, 8)
    N_FILTERS = 16

    image = imread(IMAGES_FOLDER / "sunandsea.jpg", as_gray=True)

    filters = pca_bank(image, PATCH_SIZE, N_FILTERS)

    # Get responses of 2D convolving with the first 16 images
    responses = []
    for filter in tqdm(filters, desc=f"Convolving {N_FILTERS} PCA filters"):
        responses.append(fftconvolve(image, filter, mode="same"))

    
    # Make plots
    OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)
    plot_grid(
        filters,
        OUTPUT_FOLDER / f"top_{N_FILTERS}_pc.png",
        "Top 16 PCA Learned Filters"
    )

    plot_grid(
        responses,
        OUTPUT_FOLDER / f"conv_{N_FILTERS}_responses.png",
        "Responses of Convolution with PCA Filters"
    )

    H, W = image.shape
    # responses is (16, H, W) — stack into feature matrix
    feature_matrix = np.stack(responses, axis=-1).reshape(H*W, N_FILTERS)
    # Do KMeans to get 3 clusters on the responses
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(feature_matrix)
    segmentation = labels.reshape(H, W)

    # Plot the clustering
    save_path = OUTPUT_FOLDER / "segmentation_pca.png"
    plot_kmeans_segmentation(image, segmentation, save_path)
