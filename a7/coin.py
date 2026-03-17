
if __name__ == "__main__":
    from const import IMAGES_FOLDER, OUTPUT_FOLDER
    from plotting import plot_overlay_labels
    from skimage.measure import label, regionprops
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.morphology import opening, closing, disk
    from skimage.io import imread

    # DENOMINATIONS = {
    #     # (has_hole, size_rank_within_group) -> (value_in_ore, label)
    #     # No-hole group sorted by area desc: 20kr, 50ore
    #     (False, 0): (2000, "20 kr"),
    #     (False, 1): (50, "50 øre"),
    #     # With-hole group sorted by area desc: 5kr, 2kr, 1kr
    #     (True, 0): (500, "5 kr"),
    #     (True, 1): (200, "2 kr"),
    #     (True, 2): (100, "1 kr"),
    # }

    # Load the image
    image = imread(IMAGES_FOLDER / "money_bin.jpg")
    image = closing(image, footprint=disk(1)) # Radius 1
    image = opening(image, footprint=disk(1)) # Radius 2

    # Threshold the image to create a binary image
    threshold = 128
    X = np.where(image > threshold, 0, 255)
    X_comp = np.where(image > threshold, 255, 0)

    X_labels = label(X, connectivity=2) # type: ignore
    X_comp_labels = label(X_comp, connectivity=2) # type: ignore
    X_props = regionprops(X_labels)
    X_comp_props = regionprops(X_comp_labels)

    # List of objects with holes
    holed = []
    complete = []
    for coin in X_props:
        cx1, cy1, cx2, cy2 = coin.bbox
        found = False
        for hole in X_comp_props:
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
    print(sum)
