import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from const import OUTPUT_FOLDER, IMAGES_FOLDER
from skimage.io import imread, imsave

out = OUTPUT_FOLDER / "task1"
out.mkdir(parents=True, exist_ok=True)

image = imread(IMAGES_FOLDER / "matrikelnumre_art.png")
gray_image = rgb2gray(image)

edges = canny(gray_image, sigma=2) # upon slight testing, 2 is a good sigma

# Save edges as segmented image
imsave(out / "matrikelnumre_art_canny_edges.png", edges.astype(np.uint8) * 255)
