import numpy as np 
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def hough_transform(edges:NDArray): 
    d = round(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    acc = np.zeros((2*d, 181))
    theta_rads  = [np.deg2rad(theta) for theta in range(181)]

    for x in range(edges.shape[0]): 
        for y in range(edges.shape[1]): 
            if edges[x,y] == 0: 
                pass
            else: 
                for i, theta in enumerate(theta_rads):
                    rho = round(x * np.cos(theta) + y * np.sin(theta))
                    acc[rho + d, i] += 1
    return acc

def sanity_check_hough():
    # Create simple test image (diagonal line)
    img = np.zeros((100, 200)).astype(np.uint8)
    for i in range(100):
        img[i, i] = 255  # line y = x

    acc = hough_transform(img)

    # Check peak
    max_votes = np.max(acc)
    peak = np.unravel_index(np.argmax(acc), acc.shape)

    print("Max votes:", max_votes)
    print("Peak (rho_idx, theta):", peak)

    # Visualize
    plt.imshow(acc, aspect='auto')
    plt.title("Hough Accumulator (Sanity Check)")
    plt.xlabel("theta")
    plt.ylabel("rho")
    plt.colorbar()
    plt.show()

    return acc, peak


if __name__ == "__main__":
    from const import IMAGES_FOLDER
    from skimage.feature import canny
    from skimage.io import imread

    image = imread(IMAGES_FOLDER / "cross.png")
    hough_img = hough_transform(image)


    # Check peak
    max_votes = np.max(hough_img)
    peak = np.unravel_index(np.argmax(hough_img), hough_img.shape)

    print("Max votes:", max_votes)
    print("Peak (rho_idx, theta):", peak)

    final = hough_img // max_votes * 255

    # Visualize
    plt.imshow(final, aspect="auto", cmap="gray")
    plt.title("Hough Accumulator (Sanity Check)")
    plt.xlabel("theta")
    plt.ylabel("rho")
    plt.colorbar()
    plt.show()
