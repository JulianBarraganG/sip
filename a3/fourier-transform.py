import matplotlib.pyplot as plt 
from scipy.fft import fft2, fftshift
import numpy as np
from skimage.io import imread, imsave
from const import IMAGES_FOLDER, OUTPUT_FOLDER

#Task 1 

task1_output = OUTPUT_FOLDER / "task1"
task1_output.mkdir(parents=True, exist_ok=True)

def log_abs_int(image: np.ndarray) -> np.ndarray:
    """
    Takes the log of the absolute value of the input image and scales it to [0, 255] for visualization purposes.
    """
    log_image = np.log(1 + np.abs(image)) # take the log of the absolute value
    log_image = log_image / np.max(log_image) # normalize to [0, 1] 
    log_image = (log_image * 255).astype(np.uint8) # scale to [0, 255] for saving as an image
    return log_image

def fft2_and_shift(image: np.ndarray) -> np.ndarray:
    """
    Computes the 2D Fourier Transform of the input image and shifts the zero-frequency component to the center.
    """
    fourier_transform = fft2(image)
    imsave(task1_output / "fourier_transform.png", log_abs_int(fourier_transform)) 

    
    shifted_fourier_transform = fftshift(fourier_transform)
    imsave(task1_output / "shifted.png", log_abs_int(shifted_fourier_transform))
    
    return shifted_fourier_transform

if __name__ == "__main__":
    test_img = imread(IMAGES_FOLDER / "trui.png") 
    shifted_fourier_transform = fft2_and_shift(test_img)

