import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from const import IMAGES_FOLDER, OUTPUT_FOLDER


def segmentation_otsu(image: np.ndarray) -> np.ndarray:
    grayscale = np.mean(image, axis=2).astype(np.uint8)
    threshold = threshold_otsu(grayscale)
    bin_seg = np.where(grayscale < threshold, 0, 255).astype(np.uint8)
    print(bin_seg)
    return bin_seg


if __name__ == "__main__":
    out = OUTPUT_FOLDER / "task1"
    out.mkdir(parents=True, exist_ok=True)

    for name in ["matrikelnumre_nat.png", "matrikelnumre_art.png"]:
        img = io.imread(IMAGES_FOLDER / name)
        seg = segmentation_otsu(img)
        seg_3d = arr_3d = np.stack([seg, seg,seg], axis=-1)
        segmented_img = img * seg_3d
        stem = name.replace(".png", "")
        io.imsave(out / f"{stem}_segmented.png", segmented_img)
        print(f"{name}: Otsu threshold = {threshold_otsu(np.mean(img, axis=2).astype(np.uint8)):.1f}")
