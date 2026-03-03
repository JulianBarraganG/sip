import numpy as np 
from skimage import io
import matplotlib.pyplot as plt 
from const import * 

def segmentation(image: np.ndarray, threshold: int) -> np.ndarray| None: 
    """
    Converts image to grayscale, and sets all values under threshold to 0
    """ 
    
    
    grayscale = np.mean(image, axis = 2).astype(np.uint8) 


    bin_seg = np.where(grayscale < threshold, 0, 255).astype(np.uint8)

    return bin_seg

def plot_intensity_histogram(image: np.ndarray, savepath) -> None: 
    """
    Plots the intensity histogram of the image
    """ 
    plt.hist(image.flatten(), bins = 256)
    plt.title("Intensity Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.savefig(savepath)


if __name__ == "__main__": 
    task2_out = OUTPUT_FOLDER / "task2" 
    task2_out.mkdir(parents = True, exist_ok = True)
    pillsect_img = io.imread(IMAGES_FOLDER/ "pillsect.png")
    print(f"Image shape: {pillsect_img.shape}")

    io.imsave(task2_out / "original_ps.png", pillsect_img)
    io.imsave(task2_out / "grayscale_ps.png", np.mean(pillsect_img, axis = 2).astype(np.uint8))

    thresholded_ps = segmentation(pillsect_img, 100) 
    io.imsave(task2_out / "binary_seg_100.png", thresholded_ps)

    thresholded_ps = segmentation(pillsect_img, 50) 
    io.imsave(task2_out / "binary_seg_50.png", thresholded_ps)

    thresholded_ps = segmentation(pillsect_img, 140) 
    io.imsave(task2_out / "binary_seg_140.png", thresholded_ps)


    plot_intensity_histogram(pillsect_img, task2_out / "intensity_histogram.png")
