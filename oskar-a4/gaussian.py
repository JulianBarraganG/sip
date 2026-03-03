import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from const import * 
from skimage.io import imread, imsave


def scale_fft(image, sigma): 
    """
    Apply Gaussian blur to the input image in the Fourier domain
    """

    H, W = image.shape 
    u = fftfreq(H)
    v = fftfreq(W)
    U, V = np.meshgrid(u, v, indexing='ij')
    kernel = np.exp(-2 * np.pi**2 * sigma**2 * (U**2 + V**2))

    # Apply Fourier transform to the image and kernel
    image_fft = fft2(image)
    blurred_fft = image_fft * kernel 

    # Inverse Fourier transform to get the blurred image
    blurred_image = np.real(ifft2(blurred_fft))
    return blurred_image

if __name__ == "__main__": 
    output_path = OUTPUT_FOLDER / "task4"
    output_path.mkdir(parents = True, exist_ok = True)
    trui_image = imread(IMAGES_FOLDER / "trui.png")

    imsave(output_path / "original_trui.png", trui_image.astype(np.uint8))

    blurred_trui = scale_fft(trui_image, sigma = 1)
    imsave(output_path / "blurred_trui_1.png", blurred_trui.astype(np.uint8))
    blurred_trui = scale_fft(trui_image, sigma = 2)
    imsave(output_path / "blurred_trui_2.png", blurred_trui.astype(np.uint8))

    blurred_trui = scale_fft(trui_image, sigma = 5)
    imsave(output_path / "blurred_trui_5.png", blurred_trui.astype(np.uint8))

    blurred_trui = scale_fft(trui_image, sigma = 10)
    imsave(output_path / "blurred_trui_10.png", blurred_trui.astype(np.uint8))
 



    