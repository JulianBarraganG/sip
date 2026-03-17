from skimage.color import rgb2gray
from skimage.morphology import isotropic_closing
from skimage.filters import threshold_otsu
import numpy as np

from numpy.typing import NDArray
from typing import Any

def close_bin(bin_img: NDArray[np.uint8], radius: float) -> NDArray[np.uint8]:
    return isotropic_closing(bin_img, radius)

def segmentation_otsu(image: NDArray[Any]) -> NDArray[np.uint8]:
    assert image.ndim == 2, "Input image must be grayscale"
    threshold = threshold_otsu(image)

    return np.where(image < threshold, 0, 1).astype(np.uint8)

if __name__ == "__main__":
    from const import IMAGES_FOLDER, OUTPUT_FOLDER
    from skimage.io import imread, imsave
    out = OUTPUT_FOLDER / "task4"
    out.mkdir(parents=False, exist_ok=True)

    image = imread(IMAGES_FOLDER / "matrikelnumre_nat.png")
    gray_img = rgb2gray(image)
    bin_img = segmentation_otsu(gray_img)
    closed_img = close_bin(bin_img, radius=20)
    output = (closed_img * 255).astype(np.uint8)
    imsave(out / "binary_closed.png", output)
