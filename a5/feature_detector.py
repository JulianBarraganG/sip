import numpy as np 
from numpy.typing import NDArray
from skimage.io import imread, imsave
from skimage.feature import canny, corner_harris, corner_peaks
import matplotlib.pyplot as plt
from const import OUTPUT_FOLDER, IMAGES_FOLDER


def harris_corners(image: NDArray, N: int, savepath:str) -> None: 
    
    if image.ndim != 2:
        image = np.mean(image, axis=-1)  # Convert to grayscale if it's RGB
    # Compute Harris corner response map
    harris_response = corner_harris(image, sigma= 3, k = 0.1)

    # Find local maxima (corners), keeping only the N strongest
    corners = corner_peaks(
        harris_response,
        num_peaks=N, min_distance=5
    )
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.plot(corners[:, 1], corners[:, 0], 'r+', markersize=6, markeredgewidth=1.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath)


if __name__ == "__main__": 

    output = OUTPUT_FOLDER / "task3"
    output.mkdir(exist_ok=True, parents=True)
    ###TASK 1

    task1_output = output / "task1"

    task1_output.mkdir(exist_ok=True, parents=True)

    hand_image = imread(IMAGES_FOLDER / "hand.tiff")

    imsave(output / "hand_original.png", hand_image)

    sigma = [1, 2, 3]
    low_threshold = [15, 30, 50]
    high_threshold = [40, 60, 120]

    for s in sigma:
        edges = canny(hand_image, sigma=s)
        filename = f"canny_sigma{s}.png"
        imsave(task1_output / filename, edges.astype(np.uint8) * 255)
    for low in low_threshold:
        edges = canny(hand_image, low_threshold=low)
        filename = f"canny_low{low}.png"
        imsave(task1_output / filename, edges.astype(np.uint8) * 255)
    for high in high_threshold:
        edges = canny(hand_image, high_threshold=high)
        filename = f"canny_high{high}.png"
        imsave(task1_output / filename, edges.astype(np.uint8) * 255)
               



    ###TASK 2 
    task2_output = output / "task2"
    task2_output.mkdir(exist_ok=True, parents=True)
    houses_image = imread(IMAGES_FOLDER / "modelhouses.png")
    imsave(task2_output / "houses_original.png", houses_image)
    
    sigma = [1, 2, 3]
    ks = [0.01, 0.05, 0.1]
    eps = [1e-7, 1e-6, 1e-5]

    if houses_image.ndim == 2:
        houses_rgb = np.stack([houses_image] * 3, axis=-1)
    else:
        houses_rgb = houses_image.copy()

    for s in sigma:
        c_h = corner_harris(houses_image, sigma=s, method='k')
        vis = houses_rgb.copy()
        vis[c_h > 0.01 * c_h.max()] = [255, 0, 0]  # Mark corners in red
        filename2 = f"ch_on_house_sigma{s}.png"
        imsave(task2_output / filename2, vis.astype(np.uint8))

    for k in ks:
        c_h = corner_harris(houses_image, k=k, method='k')
        vis = houses_rgb.copy()
        vis[c_h > 0.01 * c_h.max()] = [255, 0, 0]  # Mark corners in red
        filename2 = f"ch_on_house_k{k}.png"
        imsave(task2_output / filename2, vis.astype(np.uint8))

    for ep in eps: 
        c_h = corner_harris(houses_image, eps=ep, method='eps')
        vis = houses_rgb.copy()
        vis[c_h > 0.01 * c_h.max()] = [255, 0, 0]  # Mark corners in red
        filename2 = f"ch_on_house_ep{ep}.png"
        imsave(task2_output / filename2, vis.astype(np.uint8))


    ###TASK 3 

    task3_output = output / "task3"
    task3_output.mkdir(exist_ok=True, parents=True)

    harris_corners(houses_image, N=250,  savepath=task3_output / "harris_corners_N_250.png")
