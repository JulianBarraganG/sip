from skimage.morphology import opening, closing, disk
from skimage.measure import label


if __name__ == "__main__":
    from const import IMAGES_FOLDER, OUTPUT_FOLDER
    from plotting import plot_open_close, plot_overlay_labels
    from skimage.io import imread


    image_path = IMAGES_FOLDER / "cells_binary_inv.png"
    out_path = OUTPUT_FOLDER / "task4"
    out_path.mkdir(parents=False, exist_ok=True)

    image = imread(image_path) # Grayscale (H, W) image

    ### Task 4.1
    closed = closing(image, footprint=disk(1)) # Radius 1
    opened = opening(image, footprint=disk(2)) # Radius 2

    # Plot closed and original side by side
    plot_open_close(image, opened, closed, out_path / "opening_closing.png")
    # Plot zoomed in region of the three
    slice = (image.shape[0] // 2, image.shape[1] // 2)
    sliced_img = image[:slice[0], :slice[1]]
    plot_open_close(
        sliced_img,
        closed[:slice[0], :slice[1]],
        opened[:slice[0], :slice[1]],
        save_path = out_path / "zoomed_opening_closing.png"
    )

    # Task 4.1.3
    labels_closed, num_closed = label(closed, return_num=True, connectivity=2) # type: ignore
    labels_opened, num_opened = label(opened, return_num=True, connectivity=2) # type: ignore

    plot_overlay_labels(sliced_img, labels_closed[:slice[0], :slice[1]], out_path / "closed_labeled.png")
    plot_overlay_labels(sliced_img, labels_opened[:slice[0], :slice[1]], out_path / "open_labeled.png")
