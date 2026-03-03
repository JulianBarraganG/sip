import numpy as np 
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2


from const import * 

def gaussian_kernel(size: int, sigma: float) -> NDArray:
    """Generates a 2D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def camera_blur_kernel(size: int) -> NDArray:
    """Generates a simple camera blur kernel."""
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    return kernel

def lsi_degrade(image: NDArray, kernel: NDArray, noise_image: NDArray) -> NDArray:
    blurred = convolve2d(image, kernel, mode='same', boundary='wrap')
    degraded = blurred + noise_image
    return np.clip(degraded, 0, 255)


def direct_inverse_filtering(degraded_image:NDArray, kernel:NDArray) -> NDArray:
    G = fft2(degraded_image) 
    H = fft2(kernel, s=degraded_image.shape)

    if np.any(H == 0):
        print("Warning: Kernel has zero values in the frequency domain, which may lead to instability in inverse filtering.")
        H[H == 0] = 1e-10  # Avoid division by zero by adding a small constant
    F = G / H 
    restored_image = np.real(ifft2(F)) 

    return restored_image


def wiener_filtering(degraded_image:NDArray, kernel:NDArray, noise_power: float) -> NDArray:
    return None
if __name__ == "__main__": 


    ###TASK 1 

    output = OUTPUT_FOLDER / "task2"
    output.mkdir(exist_ok=True, parents=True)
    image = imread(IMAGES_FOLDER / "trui.png")

    noise_1 = np.random.normal(loc=0, scale=5, size=image.shape)
    noise_2 = np.random.normal(loc=0, scale=10, size=image.shape)
    noise_3 = np.random.normal(loc=0, scale=25, size=image.shape)

    gauss_kernel = gaussian_kernel(size=5, sigma=5)

    cam_kernel = camera_blur_kernel(size=5)
    

    degraded_1_gauss_kernel = lsi_degrade(image, gauss_kernel, noise_1)
    degraded_2_gauss_kernel = lsi_degrade(image, gauss_kernel, noise_2)
    degraded_3_gauss_kernel = lsi_degrade(image, gauss_kernel, noise_3)

    degraded_1_cam_kernel = lsi_degrade(image, cam_kernel, noise_1)
    degraded_2_cam_kernel = lsi_degrade(image, cam_kernel, noise_2)
    degraded_3_cam_kernel = lsi_degrade(image, cam_kernel, noise_3)

    imsave(output / "degraded_1_gauss.png", degraded_1_gauss_kernel.astype(np.uint8))
    imsave(output / "degraded_2_gauss.png", degraded_2_gauss_kernel.astype(np.uint8))
    imsave(output / "degraded_3_gauss.png", degraded_3_gauss_kernel.astype(np.uint8))

    imsave(output / "degraded_1_cam.png", degraded_1_cam_kernel.astype(np.uint8))
    imsave(output / "degraded_2_cam.png", degraded_2_cam_kernel.astype(np.uint8))
    imsave(output / "degraded_3_cam.png", degraded_3_cam_kernel.astype(np.uint8))


    ###TASK 2 

    restored_1_gauss = direct_inverse_filtering(degraded_1_gauss_kernel, gauss_kernel)
    restored_2_gauss = direct_inverse_filtering(degraded_2_gauss_kernel, gauss_kernel)
    restored_3_gauss = direct_inverse_filtering(degraded_3_gauss_kernel, gauss_kernel)

    restored_1_cam = direct_inverse_filtering(degraded_1_cam_kernel, cam_kernel)
    restored_2_cam = direct_inverse_filtering(degraded_2_cam_kernel, cam_kernel)
    restored_3_cam = direct_inverse_filtering(degraded_3_cam_kernel, cam_kernel)

    imsave(output / "restored_1_gauss.png", np.clip(restored_1_gauss, 0, 255).astype(np.uint8))
    imsave(output / "restored_2_gauss.png", np.clip(restored_2_gauss, 0, 255).astype(np.uint8))
    imsave(output / "restored_3_gauss.png", np.clip(restored_3_gauss, 0, 255).astype(np.uint8)) 

    imsave(output / "restored_1_cam.png", np.clip(restored_1_cam, 0, 255).astype(np.uint8))
    imsave(output / "restored_2_cam.png", np.clip(restored_2_cam, 0, 255).astype(np.uint8))
    imsave(output / "restored_3_cam.png", np.clip(restored_3_cam, 0, 255).astype(np.uint8))


    ###TASK 3 

    