import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d


def white_square(size: int) -> NDArray[np.uint8]:
    if size % 2 == 0:
        size += 1
    img = np.zeros((size, size), dtype=np.uint8)
    
    center = size // 2
    half = size // 4  # half-width of the square
    
    sq_start = center - half
    sq_end = center + half + 1  # +1 to include center pixel
    
    img[sq_start:sq_end, sq_start:sq_end] = 255

    return img

def translate_by_kernel(
    image: NDArray[np.uint8],
    x_trans: int,
    y_trans: int,
) -> NDArray[np.uint8]:
    """
    Translate the image in x and y direction. Using image coordinate convention
    where origin (0, 0) is top left.
    """
    size = max(np.abs(y_trans), np.abs(x_trans))*2 + 1
    kernel = np.zeros((size, size)).astype(np.uint8)
    kernel[size // 2 + y_trans, size // 2 + x_trans] = 1
    
    output = convolve2d(image, kernel, mode="same")

    return output

def translate_by_hc(
    image: NDArray[np.uint8],
    x_trans: float,
    y_trans: float,
) -> NDArray[np.uint8]:
    """
    Translate image in x and y direction by homogeneous coordinate 
    transformation.
    """
    phi_inv = np.identity(3)
    phi_inv[0, 2] = -x_trans
    phi_inv[1, 2] = -y_trans
    
    output = np.zeros_like(image) # I tilde

    def translate_homo_to_cart(h: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([h[0]/h[2], h[1]/h[2]]).flatten()

    def nn_interpolation(x: float, y: float) -> tuple[int, int]:
        return round(x), round(y)

    width, height = image.shape
    for y in range(height):
        for x in range(width):
            xy_coord = np.array([[x, y, 1]]).transpose()
            J_homo = phi_inv @ xy_coord
            J_cart = translate_homo_to_cart(J_homo)
            J_cart_nn = nn_interpolation(*J_cart)
            if 0 <= J_cart_nn[0] < width and 0 <= J_cart_nn[1] < height:
                output[x, y] = image[*J_cart_nn]
            
    return output

def translate_by_fft(
    image: NDArray[np.uint8],
    x_trans: float,
    y_trans: float,
):

    width, height = image.shape

    # Frequency coordinates in cycles-per-pixel, range [-0.5, 0.5)
    u = np.fft.fftfreq(width)  # horizontal frequencies, shape (cols,)
    v = np.fft.fftfreq(height)  # vertical frequencies,   shape (rows,)

    # Prepare and perform fft on image
    U, V = np.meshgrid(u, v)
    F = np.fft.fft2(image)
    translation = np.exp(-1j*2*np.pi*(U*x_trans + V*y_trans))
    F_shifted= F * translation

    # Inverse FFT; take real part to discard numerical imaginary residuals
    output = np.real(np.fft.ifft2(F_shifted))

    return output

if __name__ == "__main__":
    from const import OUTPUT_FOLDER, IMAGES_FOLDER
    from plotting import plot_translation
    from skimage.io import imread


    image = white_square(10)

    OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)

    
    X_TRANS_I = 1
    Y_TRANS_I = 2
    X_TRANS_F = 0.6
    Y_TRANS_F = 1.2

    # 1.4
    # SAVE THE WHITE SQUARE

    # 1.5
    plot_translation(
        image=image,
        x_trans=X_TRANS_I,
        y_trans=Y_TRANS_I,
        translation_function=translate_by_kernel,
        save_path=(OUTPUT_FOLDER / f"translated_{X_TRANS_I}_{Y_TRANS_I}_square_by_kernel.png"),
        title="Translated Square\n(by Kernel Translation)"
    )

    # 1.6
    plot_translation(
        image=image,
        x_trans=X_TRANS_F,
        y_trans=Y_TRANS_F,
        translation_function=translate_by_hc,
        save_path=(OUTPUT_FOLDER / f"translated_{X_TRANS_F}_{Y_TRANS_F}_square_by_hc.png"),
        title="Translated Square\n(by Homogeneous Translation)",
    )

    # 1.7
    plot_translation(
        image=image,
        x_trans=X_TRANS_I,
        y_trans=Y_TRANS_I,
        translation_function=translate_by_fft,
        save_path=(OUTPUT_FOLDER / f"translated_{X_TRANS_I}_{Y_TRANS_I}_square_by_fft.png"),
        title="Translated Square\n(by FFT Translation)",
    )

    # 1.8.1
    plot_translation(
        image=image,
        x_trans=X_TRANS_F,
        y_trans=Y_TRANS_F,
        translation_function=translate_by_fft,
        save_path=(OUTPUT_FOLDER / f"translated_{X_TRANS_F}_{Y_TRANS_F}_square_by_fft.png"),
        title="Translated Square\n(by FFT Translation)",
    )
    # 1.8.1
    real_img = imread(IMAGES_FOLDER / "trui.png")
    scale = 30
    x_trans_trui = X_TRANS_F * scale + 0.5
    y_trans_trui = Y_TRANS_F * scale + 0.5
    plot_translation(
        image=real_img,
        x_trans=x_trans_trui,
        y_trans=y_trans_trui,
        translation_function=translate_by_fft,
        save_path=(OUTPUT_FOLDER / f"translated_{x_trans_trui}_{y_trans_trui}_trui_by_fft.png"),
        title="Translated (by float) Trui\n(by FFT Translation)",
    )
    # 1.8.2
    plot_translation(
        image=real_img,
        x_trans=round(x_trans_trui),
        y_trans=round(y_trans_trui),
        translation_function=translate_by_fft,
        save_path=(OUTPUT_FOLDER / f"translated_{round(x_trans_trui)}_{round(y_trans_trui)}_trui_by_fft.png"),
        title="Translated (by int) Trui\n(by FFT Translation)",
    )
