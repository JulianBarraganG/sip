import numpy as np 
from skimage.io import imread
from scipy.signal import fftconvolve
from numpy.typing import NDArray
from pyramid_plots import plot_pyramid
from const import OUTPUT_FOLDER, IMAGES_FOLDER


def gaussian_filter_bank(size: int, sigma: NDArray[np.float64]) -> NDArray:
    """Generates a 3-jet gaussian filter bank of the specified size and sigma values."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    #base
    G = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))  # base, unnormalized
    G = G/G.sum()

    #sigma**order to ensure the same scale across different orders of derivatives. 
    #see slide 46 in scalespace.pdf 


    # 1st order
    Gx = (-xx / sigma**2) * G * sigma**1
    Gy = (-yy / sigma**2) * G * sigma**1

    # 2nd order
    Gxx = (xx**2 / sigma**4 - 1 / sigma**2) * G * sigma**2
    Gxy = (xx * yy / sigma**4) * G * sigma**2
    Gyy = (yy**2 / sigma**4 - 1 / sigma**2) * G * sigma**2

    # 3rd order
    Gxxx = (-xx**3 / sigma**6 + 3*xx / sigma**4) * G * sigma**3
    Gxxy = (-xx**2 * yy / sigma**6 + yy / sigma**4) * G * sigma**3
    Gxyy = (-xx * yy**2 / sigma**6 + xx / sigma**4) * G * sigma**3
    Gyyy = (-yy**3 / sigma**6 + 3*yy / sigma**4) * G * sigma**3

    # Normalize each kernel to have a sum of 1

    return [G, Gx, Gy, Gxx, Gxy, Gyy, Gxxx, Gxxy, Gxyy, Gyyy]

def make_and_plot_filter_bank(size:int, sigma:int,labels:list)-> list: 
    filters = gaussian_filter_bank(size, sigma)

    plot_pyramid(
        images = filters,
        labels = labels,
        title = fr"3-jet Gaussian filter bank for $\sigma={sigma}$",
        savepath = OUTPUT_FOLDER / f"gaussian_filter_bank_sigma_{sigma}.png"
    )

    return filters


def apply_filter_bank_and_plot(
    image: NDArray, filter_bank: list, labels: list, sigma:int
) -> list:
    filtered_images = []
    for kernel in tqdm(filter_bank, desc="Filtering with N-Jet filters"):
        filtered = fftconvolve(image, kernel, mode="same")
        filtered_images.append(filtered)

    plot_pyramid(
        images = filtered_images,
        labels = labels,
        title = fr"Filtered images with 3-jet Gaussian filter bank $\sigma=${sigma}",
        savepath = OUTPUT_FOLDER / f"filtered_images_sigma_{sigma}.png"
    )

    return filtered_images


if __name__ == "__main__":
    from plotting import plot_kmeans_segmentation
    from tqdm import tqdm
    SIZE = 25 
    LABELS = [
        [r"$G$"],
        [
            r"$\frac{\partial}{\partial x} G$",
            r"$\frac{\partial}{\partial y} G$",
        ],
        [
            r"$\frac{\partial}{\partial x^2} G$",
            r"$\frac{\partial}{\partial xy} G$",
            r"$\frac{\partial}{\partial y^2} G$",
        ],
        [
            r"$\frac{\partial}{\partial x^3} G$",
            r"$\frac{\partial}{\partial x^2y} G$",
            r"$\frac{\partial}{\partial xy^2} G$",
            r"$\frac{\partial}{\partial y^3} G$",
        ]
    ]
    
    sigma_values = [1, 3, 5] 
    filter_banks = []
    for sigma in sigma_values:
        filter_banks.append(make_and_plot_filter_bank(SIZE, sigma, LABELS))

    all_responses = {}
    for idx, filter_bank in enumerate(filter_banks):
        image = imread(IMAGES_FOLDER / "sunandsea.jpg", as_gray=True)
        all_responses[sigma_values[idx]] = apply_filter_bank_and_plot(
            image, filter_bank, LABELS, sigma_values[idx]
        )


    # Part 3.3
    from sklearn.cluster import KMeans
    import numpy as np

    # Use only sigma=5 filter bank (last one in filter_banks, since sigma_values = [1, 3, 5])
    image = imread(IMAGES_FOLDER / "sunandsea.jpg", as_gray=True)

    # Build feature matrix (H*W, 10)
    H, W = image.shape
    feature_matrix = np.stack(all_responses[sigma_values[-1]], axis=-1).reshape(H * W, -1)

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(feature_matrix)
    segmentation = labels.reshape(H, W)

    plot_kmeans_segmentation(
        image,
        segmentation,
        save_path=OUTPUT_FOLDER / "segmentation_njet.png",
        title="N-Jet Filter Bank Segmentation (K=3, σ=5)"
    )
