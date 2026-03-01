import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from numpy.typing import NDArray
from typing import Any


def scale_fft(
    image: NDArray[Any],
    sigma: float,
) -> NDArray[np.float64]:
    """Convolve an image with a Gaussian kernel via multiplication in the Fourier domain.

    The Gaussian kernel is constructed directly in frequency space using the
    analytical Fourier transform of the Gaussian. Given a spatial Gaussian
    parameterised by standard deviation σ, the corresponding Fourier-domain
    filter is:

        G(u, v) = exp(-2π²σ²(u² + v²))

    which follows from substituting a = 1/(2σ²) into the result derived on
    the lecture slide: G(u) = exp(-π²u²/a).

    Parameters
    ----------
    image:
        2-D grayscale image array. Values are expected in [0, 255] or [0, 1].
    sigma:
        Standard deviation of the Gaussian kernel in the spatial domain
        (units: pixels).

    Returns
    -------
    NDArray[np.float64]
        Smoothed image with the same shape as the input.
    """
    assert sigma > 0, "sigma must be positive"
    assert image.ndim == 2, "image must be a 2-D grayscale array"

    rows, cols = image.shape

    # Frequency coordinates in cycles-per-pixel, range [-0.5, 0.5)
    u = np.fft.fftfreq(cols)  # horizontal frequencies, shape (cols,)
    v = np.fft.fftfreq(rows)  # vertical frequencies,   shape (rows,)

    # Prepare and perform fft on image
    U, V = np.meshgrid(u, v)
    F = np.fft.fft2(image)
    # Multiply in frequency space with Gaussian filter kernel function G
    G = np.exp(-2 * np.pi**2 * sigma**2 * (U**2 + V**2))
    F_filtered = F * G # elementwise multiplication i.e. broadcasting

    # Inverse FFT; take real part to discard numerical imaginary residuals
    smoothed = np.real(np.fft.ifft2(F_filtered))

    return smoothed


if __name__ == "__main__":
    from skimage.io import imread
    from skimage.data import chelsea  # fallback if trui.png is unavailable
    from pathlib import Path

    DATA_FOLDER = Path("data") / "images"
    OUTPUT_FOLDER = Path(__name__).parent / "julian-a4" / "output"
    OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)

    # Load trui.png; fall back to a bundled skimage image if file is missing
    trui_path = DATA_FOLDER / "trui.png"
    raw = imread(trui_path)
    image = rgb2gray(raw) if raw.ndim == 3 else raw.astype(np.float64)

    SIGMAS = [1, 3, 7, 15, 30]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(SIGMAS) + 1,
        figsize=(3 * (len(SIGMAS) + 1), 3.5),
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, sigma in zip(axes[1:], SIGMAS):
        smoothed = scale_fft(image, sigma=sigma)
        ax.imshow(smoothed, cmap="gray", vmin=image.min(), vmax=image.max())
        ax.set_title(f"σ = {sigma}")
        ax.axis("off")

    fig.suptitle("Gaussian smoothing via Fourier-domain multiplication", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "scale_fft.png", bbox_inches="tight", dpi=150)
    plt.close()
