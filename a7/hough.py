import numpy as np 
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from skimage.transform import hough_line, hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage.feature import canny 
from skimage.morphology import closing, disk
from skimage.filters import threshold_otsu 
from pathlib import Path

def hough_transform(edges:NDArray): 
    d = round(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    acc = np.zeros((2*d, 180))
    theta_rads  = [np.deg2rad(theta) for theta in range(-90,90)]

    for row in range(edges.shape[0]):
      for col in range(edges.shape[1]):
          if edges[row, col] != 0:
              for i, theta in enumerate(theta_rads):
                  rho = round(col * np.cos(theta) + row * np.sin(theta))
                  acc[rho + d, i] += 1
    return acc

def sanity_check_hough():
    # Create simple test image (diagonal line)
    img = np.zeros((100, 200)).astype(np.uint8)
    for i in range(100):
        img[i, i] = 255  # line y = x

    acc = hough_transform(img)
    # Visualize
    plt.imshow(acc, aspect='auto', cmap='gray')
    plt.title("Hough Accumulator (Sanity Check)")
    plt.xlabel("theta")
    plt.ylabel("rho")
    tick_positions = np.linspace(0, 179, 7)
    tick_labels = [f"{int(t - 90)}" for t in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    plt.colorbar()
    plt.show()
    plt.close()
    return acc


def circle(image:NDArray, radius:int, save_path:Path):
    grayscale_img = rgb2gray(image) 
    th_img = np.where(grayscale_img > threshold_otsu(grayscale_img), 255, 0)
    closed_img = closing(th_img, disk(10)).astype(np.uint8)

    edges = canny(closed_img, sigma= 3)
    circ_acc = hough_circle(image=edges, radius=radius)
    _, cx, cy, radii = hough_circle_peaks(circ_acc, [radius], num_peaks=1)
    result = image.copy()
    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y = np.clip(circy + dy, 0, result.shape[0] - 1)
                x = np.clip(circx + dx, 0, result.shape[1] - 1)
                result[y, x] = (220, 20, 20)

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            result[cy+dy, cx + dx] = (220, 20, 20)
    ax.imshow(result, cmap="gray")
    fig.savefig(save_path)
    plt.close()

def plot_hough(acc, save_path): 

    plt.imshow(acc, aspect='auto', cmap='gray')
    plt.title("Hough Space")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\rho$")
    tick_positions = np.linspace(0, 179, 7)
    tick_labels = [f"{int(t - 90)}" for t in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close() 

def detect_and_plot_lines(image: NDArray, acc: NDArray, num_lines: int, save_path):
    d = round(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    
    # Find peaks in accumulator
    detected_lines = []
    acc_copy = acc.copy()
    for _ in range(num_lines):
        idx = np.argmax(acc_copy)
        rho_idx, theta_idx = np.unravel_index(idx, acc_copy.shape)
        rho = rho_idx - d
        theta = np.deg2rad(theta_idx - 90)
        detected_lines.append((rho, theta))
        # Suppress peak neighborhood to avoid duplicates
        acc_copy[rho_idx-10:rho_idx+10, theta_idx-10:theta_idx+10] = 0

    # Draw lines on image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for rho, theta in detected_lines:
        # Convert (rho, theta) to two points
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        # Extend line across image using a large t
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    from const import IMAGES_FOLDER, OUTPUT_FOLDER
    from skimage.feature import canny
    from skimage.io import imread
    

    out_path = OUTPUT_FOLDER / "task3"
    out_path.mkdir(parents=False, exist_ok= True)
    image = imread(IMAGES_FOLDER / "cross.png")
    hough_img = hough_transform(image)
    plot_hough(hough_img, out_path / "cross_hs.png") 
    detect_and_plot_lines(image, hough_img, num_lines=2, save_path=out_path / "cross_lines.png")

    sk_hough, _, _ = hough_line(image) 
    plot_hough(sk_hough, out_path / "sk_cross_hs.png")

    rot_img = imread(OUTPUT_FOLDER/"task2"/"rotated_map.png") 
    circle(rot_img[:,:,:3], radius=30, save_path=out_path / "circle_detection.png")
