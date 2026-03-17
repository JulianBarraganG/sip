import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np


def plot_open_close(original, closed, opened, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(closed, cmap="gray")
    axes[1].set_title("Closed Image")
    axes[1].axis("off")
    axes[2].imshow(opened, cmap="gray")
    axes[2].set_title("Opened Image")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_overlay_labels(image, labels, save_path):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    # For each unique label (skip 0 if it's background)
    for label_id in np.unique(labels):
          if label_id == 0:
              continue

          # Find centroid of the label region
          mask = labels == label_id
          cy, cx = ndimage.center_of_mass(mask)

          ax.text(cx, cy, str(label_id),
                  color="red", fontsize=10, fontweight="bold",
                  ha="center", va="center",
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
