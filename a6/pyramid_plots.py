import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_pyramid(images: list, labels: list, title: str, savepath: Path, cmap: str = "gray") -> None:
    """
    Plots 10 images in a pyramid layout:
        Row 0:  images[0]
        Row 1:  images[1], images[2]
        Row 2:  images[3], images[4], images[5]
        Row 3:  images[6], images[7], images[8], images[9]
    Each image is labeled with the corresponding label from the labels list.
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 8)

    start_cols = [3, 2, 1, 0]

    img_idx = 0
    for row, (n_cols, start_col) in enumerate(zip([1, 2, 3, 4], start_cols)):
        for i in range(n_cols):
            col = start_col + i * 2
            ax = fig.add_subplot(gs[row, col:col+2])
            ax.imshow(images[img_idx], cmap=cmap)
            #set title under image 
            ax.axis("off")
            ax.text(0.5, -0.15, labels[row][i], fontsize=14, ha='center', transform=ax.transAxes)
            img_idx += 1

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()