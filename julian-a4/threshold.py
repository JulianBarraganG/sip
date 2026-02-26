import numpy as np
from numpy.typing import NDArray
from typing import Any

def threshold_image(
    image: NDArray[Any | np.uint8],
    threshold: int,
) -> NDArray[np.uint8]:
    """Convert image to grayscale, apply threshoding and return binary 
    segmentation mask."""
    assert threshold >= 0 and threshold <= 255, "threshold must be in [0,255] range"
    # Convert to grayscale
    if image.ndim == 3 and image.shape[2] == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    # Apply thresholding
    binary_mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    
    return binary_mask


if __name__ == "__main__":
    from skimage.io import imread
    from pathlib import Path
    from plotting import plot_thresholding

    DATA_FOLDER = Path(__name__).parent / "data"
    OUTPUT_FOLDER = Path(__name__).parent / "julian-a4" / "output"
    OUTPUT_FOLDER.mkdir(parents=False, exist_ok=True)
    IMAGES_FOLDER = DATA_FOLDER / "images"
    SAVE_PATH = OUTPUT_FOLDER / "thresholding.png"

    test_img = imread(IMAGES_FOLDER / "pillsect.png") 
    plot_thresholding(test_img, SAVE_PATH)
    # Plot the histogram using matplotlib.pyplot.hist of original image
    import matplotlib.pyplot as plt
    plt.hist(test_img.ravel(), bins=256, range=(0, 255))
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    plt.grid()
    plt.savefig(OUTPUT_FOLDER / "histogram.png")
    plt.close()
