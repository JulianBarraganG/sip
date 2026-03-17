import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

from numpy.typing import NDArray
from typing import Any


def segmentation_otsu(image: NDArray[Any]) -> NDArray[Any]:
    assert image.ndim == 2, "Input image must be grayscale"
    threshold = threshold_otsu(image)
    bin_seg = np.where(image < threshold, 0, 1)

    return bin_seg * image


if __name__ == "__main__":
    from const import IMAGES_FOLDER, OUTPUT_FOLDER
    from plotting import plot_histogram, plot_w_and_wo_color
    out = OUTPUT_FOLDER / "task1"
    out.mkdir(parents=True, exist_ok=True)

    for name in ["matrikelnumre_nat.png", "matrikelnumre_art.png"]:
        img = io.imread(IMAGES_FOLDER / name)
        stem = name.replace(".png", "")
        title = stem.replace("_", " ").capitalize()
        threshold = threshold_otsu(rgb2gray(img)) * 255
        gray_img = rgb2gray(img)
        gray_img_u8 = (gray_img * 255).astype(np.uint8)
        plot_histogram(gray_img_u8, stem, out / f"{stem}_histogram.png")
        plot_histogram(gray_img_u8, stem, out / f"{stem}_histogram_w_thresh.png", threshold)
        plot_w_and_wo_color(img, title, out / f"{stem}_derp_plot.png")
        segmented_img = (segmentation_otsu(gray_img) * 255).astype(np.uint8)
        io.imsave(out / f"{stem}_segmented.png", segmented_img)
        print(f"{name}: Otsu threshold = {threshold:.2f}")
