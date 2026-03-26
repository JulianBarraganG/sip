from typing import Any
import numpy as np
from numpy.typing import NDArray
from skimage.transform import warp


def _translate_t(t: tuple[float, ...], ndim: int = 2) -> NDArray[Any]:
    assert len(t) == ndim, "t should have the same number of dimensions as ndim"
    Tt = np.identity(ndim + 1)
    for i in range(len(t)):
        Tt[i, ndim] = t[i]

    return Tt

def _rotate(theta: float, ndim: int = 2) -> NDArray[Any]:
    """Returns the rotation matrix for a given angle theta in radians."""
    assert ndim == 2, "`_rotate` is only defined for 2D images"
    R = np.identity(ndim + 1)
    R[0, 0] = np.cos(theta)
    R[0, 1] = np.sin(theta)
    R[1, 0] = -np.sin(theta)
    R[1, 1] = np.cos(theta)

    return R

def trs(
    image: NDArray[Any],
    t: tuple[float, float],
    theta: float,
    scale: float,
) -> NDArray[Any]:
    """
    Applies the transformation $T_tT_cRST_c^{-1}$ on the input image.
    Input image must have uneven height and widt, such that it has a center 
    pixel. The input image must also be 2D grayscale image.
    """
    H, W = image.shape
    assert image.ndim == 2, "Input image must be 2D grayscale image"
    assert len(t) == 2, "t should be a tuple of length 2"
    assert scale > 0, "scale should be a positive number"
    assert H % 2 == 1, "Input image must have uneven height"
    assert W % 2 == 1, "Input image must have uneven width"
    
    c = (W // 2, H // 2)  # center of the image

    # We give the transformation product in its inverse form,
    # because `skimage.transform.warp` applies the inverse of the given transformation.
    Tt_inv = _translate_t((-t[0], -t[1]))
    Tc_inv = _translate_t((-c[0], -c[1]))
    Tc = _translate_t(c)

    R_inv = _rotate(-theta)
    S_inv = np.diag([1/scale, 1/scale, 1])

    inv_transformation = Tc @ S_inv @ R_inv @ Tc_inv @ Tt_inv

    return warp(image, inv_transformation, order=0)


if __name__ == "__main__":
    from a6.transform import white_square
    import matplotlib.pyplot as plt
    from const import OUTPUT_FOLDER

    SIZE = 201
    sq_img = white_square(SIZE)

    # Apply the function on the square image
    transformed_img = trs(sq_img, t=(10.4, 15.7), theta=np.pi/10, scale=2)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(transformed_img, cmap="gray")
    plt.title(r"Transformation: $T_tT_cRST_c^{-1}$ on White Square Image")
    plt.axis("off")
    plt.savefig(OUTPUT_FOLDER / "transformed image")
    plt.close()

    print(r"\n")
