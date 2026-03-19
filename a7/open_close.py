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
    closed = closing(image, footprint=disk(2)) # Radius 1
    opened = opening(image, footprint=disk(2)) # Radius 2

    # Plot closed and original side by side
    plot_open_close(
        original=image,
        closed=closed,
        opened=opened,
        save_path=(out_path / "opening_closing.png")
    )
    # Plot zoomed in region of the three
    slice = (image.shape[0] // 2, image.shape[1] // 2)
    sliced_img = image[:slice[0], :slice[1]]
    plot_open_close(
        original=sliced_img,
        closed=closed[:slice[0], :slice[1]],
        opened=opened[:slice[0], :slice[1]],
        save_path=(out_path / "zoomed_opening_closing.png")
    )

    # Task 4.1.3
    labels_closed, num_closed = label(closed, return_num=True, connectivity=2) # type: ignore
    labels_opened, num_opened = label(opened, return_num=True, connectivity=2) # type: ignore

    plot_overlay_labels(
        image=closed,
        labels=labels_closed,
        save_path=(out_path / "closed_labeled.png"),
        fontsize=16,
    )
    plot_overlay_labels(
        image=opened,
        labels=labels_opened,
        save_path=out_path / "opened_labeled.png",
    )

    plot_overlay_labels(
        image=closed[:slice[0], :slice[1]],
        labels=labels_closed[:slice[0], :slice[1]],
        save_path=(out_path / "zoomed_closed_labeled.png"),
        fontsize=24,
    )
    plot_overlay_labels(
        image=opened[:slice[0], :slice[1]],
        labels=labels_opened[:slice[0], :slice[1]],
        save_path=out_path / "zoomed_opened_labeled.png",
        fontsize=24,
    )

    print(f"Number of connected components in closed image: {num_closed}")
    print(f"Number of connected components in opened image: {num_opened}")
