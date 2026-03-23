from const import IMAGES_FOLDER
from skimage.measure import label, regionprops
import numpy as np
from skimage.morphology import opening, closing, disk
from skimage.io import imread

# Load the image
image = imread(IMAGES_FOLDER / "money_bin.jpg")
closed = closing(image, footprint=disk(1)) # Radius 1
opened_closed = opening(closed, footprint=disk(1)) # Radius 2

# Threshold the image to create a binary image
image = opened_closed
threshold = 128
X = np.where(image > threshold, 255, 0) # binary of original
X_comp = np.where(image > threshold, 0, 255) # white coins on black background

X_labels = label(X, connectivity=2) # type: ignore
X_comp_labels = label(X_comp, connectivity=2) # type: ignore
X_props = regionprops(X_labels)
X_comp_props = regionprops(X_comp_labels)
print(f"Len X props: {len(X_props)}, len X comp props: {len(X_comp_props)}")

# List of objects with holes
holed = []
complete = []
for coin in X_comp_props:
    cx1, cy1, cx2, cy2 = coin.bbox
    found = False
    for hole in X_props:
        hx1, hy1, hx2, hy2 = hole.bbox
        if cx1 < hx1 and cx2 > hx2 and cy1 < hy1 and cy2 > hy2:
            holed.append(coin.area)
            found = True
    if not found:
        complete.append(coin.area)

holed = np.sort(holed)
complete = np.sort(complete)

def split_two(arr):
    gaps = [arr[i+1] - arr[i] for i in range(len(arr) - 1)] # type: ignore
    split_index = gaps.index(max(gaps)) + 1

    return arr[:split_index], arr[split_index:]

halvtreds_ore, tyvere = split_two(complete)

def split_three(arr):
    gaps = [arr[i+1] - arr[i] for i in range(len(arr) - 1)] # type: ignore
    split_indices = sorted(
            sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)[:2]
        )

    # Slice into three parts
    i, j = split_indices[0] + 1, split_indices[1] + 1

    return arr[:i], arr[i:j], arr[j:]

en, to, fem = split_three(holed)

# count the sum
sum = len(halvtreds_ore) * 0.5 + len(tyvere) * 20 + len(en) + len(to) * 2 + len(fem) * 5
print(f"Total sum: {sum} DKK")
