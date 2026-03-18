import numpy as np 
from numpy.typing import NDArray
from skimage.io import imread, imsave 
from skimage.feature import corner_harris, corner_peaks 
from numpy.typing import NDArray 
from skimage.morphology import closing, disk
from const import OUTPUT_FOLDER, IMAGES_FOLDER 
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu 
from skimage.transform import ProjectiveTransform, warp
from pathlib import Path 



def four_corners(image: NDArray, save_path: Path) -> NDArray:  
    grayscale_img = rgb2gray(image) 
    th_img = np.where(grayscale_img > threshold_otsu(grayscale_img), 255, 0)
    closed_img = closing(th_img, disk(10))
    
    all_corners = corner_harris(closed_img)
    best_corners = corner_peaks(all_corners, num_peaks=4)
    _, ax = plt.subplots()
    ax.imshow(image)
    ax.plot(best_corners[:, 1], best_corners[:, 0], 'r+', markersize=15, markeredgewidth=2)
    
    plt.savefig(save_path) 
    plt.close()
    return best_corners


def homography_transform(img:NDArray, corners:NDArray, save_path: Path):
    # corners are (row, col); convert to (x, y) = (col, row)
    pts = corners[:,::-1]
    # Order: top-left, top-right, bottom-right, bottom-left
    center = pts.mean(axis=0)
    def angle(p):
        return np.arctan2(p[1] - center[1], p[0] - center[0])
    ordered = np.array(sorted(pts, key=angle))  # sorted by angle around center
    # arctan2 in image coords (y down): TL~-3pi/4, TR~-pi/4, BR~+pi/4, BL~+3pi/4
    # sorted ascending gives [TL, TR, BR, BL]
    tl, tr, br, bl = ordered[0], ordered[1], ordered[2], ordered[3]
    src = np.array([tl, tr, br, bl])

    H, W = img.shape[:2]
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=float)

    tform = ProjectiveTransform.from_estimate(dst, src)
    tf_img = warp(img, tform, preserve_range=True, output_shape=(H, W))
    imsave(save_path, tf_img.astype(np.uint8)) 

if __name__ == "__main__": 
    out_path = OUTPUT_FOLDER / "task2" 
    out_path.mkdir(parents=False, exist_ok=True) 

    img = imread(IMAGES_FOLDER/"matrikelnumre_nat.png")

    corners = four_corners(img, out_path / "map_with_corners.png")
    homography_transform(img, corners, out_path / "rotated_map.png")
