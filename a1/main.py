import numpy as np
from skimage.io import imread
from pathlib import Path
from contrast import gamma_transform_uint8, gamma_transform_value_uint8
import matplotlib.pyplot as plt
from const import DATA_FOLDER, OUTPUT_FOLDER

def main():

    image_folder = DATA_FOLDER / "images"
    cameraman_img = imread(image_folder / "cameraman.tif") 
    autumn_img = imread(image_folder / "autumn.tif") 

    for k, i in enumerate(np.arange(0.05, 2.1, 0.15)):
        i = np.round(i, decimals=2)
        camera_transformed = gamma_transform_uint8(cameraman_img, i)
        autumn_transformed = gamma_transform_uint8(autumn_img, i)
        val_autumn_transformed = gamma_transform_value_uint8(autumn_img, i)
        camera_save_path = OUTPUT_FOLDER / f"cameraman{k+1}_gamma_{i}.png"
        autumn_save_path = OUTPUT_FOLDER / f"autumn{k+1}_gamma_{i}.png"
        val_autumn_save_path = OUTPUT_FOLDER / f"val_corrected_autumn{k+1}_gamma_{i}.png"
        # If output folder doesn't exist, create it.
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        # Save transformed grayscale cameraman image
        plt.imsave(camera_save_path, camera_transformed, format="png", cmap="gray")
        # Save channel-wise gamma corrected autumn image.
        plt.imsave(autumn_save_path, autumn_transformed, format="png")
        # Save HSV Value-channel corrected autumn image.
        plt.imsave(val_autumn_save_path, val_autumn_transformed, format="png")


if __name__ == "__main__":
    main()
