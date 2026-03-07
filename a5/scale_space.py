import numpy as np
import matplotlib.pyplot as plt
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
    F_filtered = F * G  # elementwise multiplication i.e. broadcasting

    # Inverse FFT; take real part to discard numerical imaginary residuals
    smoothed = np.real(np.fft.ifft2(F_filtered))
    return smoothed


def make_gaussian_blob(
    sigma: float,
    grid_half_width: int = 50,
    grid_size: int = 201,
) -> NDArray[np.float64]:
    """Sample the 2-D Gaussian G(x, y, σ) = 1/(2πσ²) · exp(-(x²+y²)/(2σ²))
    on a discrete pixel grid centred at the origin.

    Parameters
    ----------
    sigma:
        Standard deviation of the Gaussian (spatial units = pixels).
    grid_half_width:
        Half-width of the spatial axis in pixels. The grid runs from
        -grid_half_width to +grid_half_width inclusive.
    grid_size:
        Total number of samples along each axis. Should be odd so that
        the origin is exactly a sample point.

    Returns
    -------
    x, y, blob
        x and y are 1-D coordinate arrays of shape (grid_size,).
        blob is a 2-D image of shape (grid_size, grid_size) containing the 
        sampled Gaussian values.
    """
    assert sigma > 0, "sigma must be positive"
    assert grid_size % 2 == 1, "grid_size should be odd so the origin is centred"

    x = np.linspace(-grid_half_width, grid_half_width, grid_size)
    y = np.linspace(-grid_half_width, grid_half_width, grid_size)
    X, Y = np.meshgrid(x, y)

    blob = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(
        -(X**2 + Y**2) / (2.0 * sigma**2)
    )
    return blob


if __name__ == "__main__":
    from const import OUTPUT_FOLDER
    from plotting import plot_tau_range

    OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)

    # Parameters
    SIGMA_B = 1.0
    TAU = 2.0
    SIGMA_COMBINED = np.sqrt(SIGMA_B**2 + TAU**2)

    GRID_HALF_EXTENT = 20
    GRID_SIZE = 201   # odd. origin is exactly centred
    TAU_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]

    # Part 1 – construct the base blob B(x, y) = G(x, y, σ=1)
    B = make_gaussian_blob(
        sigma=SIGMA_B,
        grid_half_width=GRID_HALF_EXTENT,
        grid_size=GRID_SIZE,
    )

    # Part 2 – scale-space: convolve B with G(τ) for several τ values
    # Illustrates I(x, y, τ) = B(x, y) * G(x, y, τ)
    scale_space_images = [scale_fft(B, sigma=tau) for tau in TAU_VALUES]
    plot_tau_range(
        blob=B,
        taus=TAU_VALUES,
        grid_half_width=GRID_HALF_EXTENT,
        scale_space_images=scale_space_images,
        sigma=SIGMA_B,
    )

    # Part 3 – verify eq. (1): G(σ) * G(τ) = G(√(σ²+τ²))                #
    #   A = B * G(τ=2)       [convolve the σ=1 blob with τ=2 Gaussian]    #
    #   C = G(x,y, √5)       [directly sampled combined blob]             #
    #   diff = A − C         [should be ≈ 0 up to discretisation error]   #
    # ------------------------------------------------------------------ #
    A = make_gaussian_blob(  # re-use helper; overwrite blob value
        sigma=SIGMA_B,
        grid_half_width=GRID_HALF_EXTENT,
        grid_size=GRID_SIZE,
    )
    A = scale_fft(B, sigma=TAU)                    # B convolved with G(τ=2)

    C = make_gaussian_blob(                  # directly sampled G(√5)
        sigma=SIGMA_COMBINED,
        grid_half_width=GRID_HALF_EXTENT,
        grid_size=GRID_SIZE,
    )

    diff = A - C

    print(f"Max absolute difference |A − C|: {np.abs(diff).max():.2e}")
    print(f"Mean absolute difference |A − C|: {np.abs(diff).mean():.2e}")

    # Shared colour scale for A and C so they are visually comparable
    vmin = min(A.min(), C.min())
    vmax = max(A.max(), C.max())

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    extent = [-GRID_HALF_EXTENT, GRID_HALF_EXTENT, -GRID_HALF_EXTENT, GRID_HALF_EXTENT]

    im0 = axes[0].imshow(A, cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title(f"A = B(σ={SIGMA_B}) * G(τ={TAU})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(C, cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title("C = G(x,y, √(σ²+τ²))")
    axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        diff, cmap="gray", vmin=vmin, vmax=vmax, extent=extent
    )
    axes[2].set_title("Difference A − C")
    axes[2].set_xlabel("x")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(
        "Verification of G(σ) ∗ G(τ) = G(√(σ²+τ²))  [eq. 1]",
        y=1.02,
    )
    for i in range(3):
        # remove x and y axis ticks 
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "4_1_verification.png", bbox_inches="tight", dpi=150)
    plt.close()
