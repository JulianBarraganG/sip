import matplotlib.pyplot as plt 
from scipy.fft import fft2, fftshift, ifft2, fftfreq
import numpy as np
from skimage.io import imread, imsave
import cv2
from const import IMAGES_FOLDER, OUTPUT_FOLDER, a0, v0, w0
from pathlib import Path

def log_plot(data: np.ndarray, title: str, x_label: str, y_label:str, savepath: Path, x_values: np.ndarray|None = None, xlim: tuple|None = None , ylim: tuple|None = None) -> None:
    plt.figure(figsize=(12, 6))
    if x_values is not None:
        plt.plot(x_values, data)
    else:
        plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.yscale("log")

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(savepath / title.replace(" ", "_")) 
    plt.close()


#Task 1 

task1_output = OUTPUT_FOLDER / "task1"
task1_output.mkdir(parents=True, exist_ok=True)

def visualize_abs_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Takes the log of the absolute value of the input image and scales it to [0, 255] for visualization purposes.
    """
    log_image = np.log(1 + np.abs(image)) # take the log of the absolute value
    log_image = log_image / np.max(log_image) # normalize to [0, 1] 
    log_image = (log_image * 255).astype(np.uint8) # scale to [0, 255] for saving as an image
    return log_image

def visualize_power_spectrum(fft: np.ndarray) -> np.ndarray:
    """
    Takes the log of the power spectrum of the input image and scales it to [0, 255] for visualization purposes.
    """
    power_spectrum = np.abs(fft) ** 2
    log_power_spectrum = np.log(1 + power_spectrum) # take the log of the power spectrum
    log_power_spectrum = log_power_spectrum / np.max(log_power_spectrum) # normalize to [0, 1] 
    log_power_spectrum = (log_power_spectrum * 255).astype(np.uint8) # scale to [0, 255] for saving as an image
    return log_power_spectrum


def power_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Computes the power spectrum of the input image.
    """
    fourier_transform = fft2(image)
    fourier_transform = fftshift(fourier_transform)
    power_spectrum = np.abs(fourier_transform) ** 2
    return power_spectrum

def fft2_and_shift(image: np.ndarray, title:str) -> np.ndarray:
    """
    Computes the 2D Fourier Transform of the input image and shifts the zero-frequency component to the center.
    """
    fourier_transform = fft2(image)
    
    shifted_fourier_transform = fftshift(fourier_transform)
    
    return shifted_fourier_transform

#Task 2 
task2_output = OUTPUT_FOLDER / "task2"
task2_output.mkdir(parents=True, exist_ok=True)

def add_cos_wave(image: np.ndarray, title: str) -> np.ndarray:
    """
    Adds a cosinus wave to the pixel values of the image.
    """
    x = np.arange(image.shape[0])  # x = column indices
    y = np.arange(image.shape[1])  # y = row indices
    xx, yy = np.meshgrid(x, y)

    wave = a0 * np.cos(v0 * xx + w0 * yy)

    modified_image = image + wave
    modified_image = np.clip(modified_image, 0, 255).astype(np.uint8) # clip values to [0, 255] and convert to uint8
    return modified_image

def filter_planar_waves(waved_image: np.ndarray, title: str, v0:int, w0:int) -> np.ndarray:
    """
    Removes the planar waves from the power spectrum
    """
    x = np.arange(waved_image.shape[0])  # x = column indices
    y = np.arange(waved_image.shape[1])  # y = row indices
    xx, yy = np.meshgrid(x, y)

    wave = np.cos(v0 * xx + w0 * yy) 
    wave_fft = fft2(wave)

    waved_image_fft = fft2(waved_image)
    magnitude = np.abs(wave_fft)
    coordinates = np.where(magnitude == np.max(magnitude))
    print(coordinates)

    radius = 6

    #set values in the power spectrum to 0 within a radius around the coordinates of the maximum value in the wave_fft
    for i in range(len(coordinates[0])):
        py, px = coordinates[0][i], coordinates[1][i]
        dist2 = (yy - py)**2 + (xx - px)**2
        waved_image_fft[dist2 <= radius**2] = 0
    fixed_image = np.real(ifft2(waved_image_fft)).astype(np.uint8)

    imsave(task2_output / (title + "_wave_removed.png"), fixed_image)

    return fixed_image


#task 3 
task3_output = OUTPUT_FOLDER / "task3"
task3_output.mkdir(parents=True, exist_ok=True)


def radial_average(shifted_power_spectrum: np.ndarray) -> np.ndarray:
    """
    Computes the radial average of the power spectrum.
    """
    H, W = shifted_power_spectrum.shape
    #get coordinates for center of the image
    center_y = H // 2
    center_x = W // 2

    x = np.arange(W) - center_x
    y = np.arange(H) - center_y
    xx, yy = np.meshgrid(x, y)

    #make an array of the same shape as the power spectrum where each value is the distance from the center of the image
    dist = np.sqrt(xx**2 + yy**2)


    smallest_dimension = min(H, W)
    largest_radius = smallest_dimension // 2 
    radial_means = np.zeros(largest_radius)
    for r in range(1,largest_radius+1):
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        radial_means[r-1] = np.mean(shifted_power_spectrum[mask])


    
    return radial_means


#Task 4 
task4_output = OUTPUT_FOLDER / "task4"
task4_output.mkdir(parents=True, exist_ok=True)


def visualize_angular_bins(shape: tuple = (300,300), bin_size: int = 10, 
                            freq_min: int = 10, freq_max: int = 100):
    H, W = shape
    cy, cx = H // 2, W // 2

    x = np.arange(W) - cx
    y = np.arange(H) - cy
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt(xx**2 + yy**2)
    angles = np.degrees(np.arctan2(yy, xx)) % 360

    freq_mask = (dist >= freq_min) & (dist <= freq_max)

    # Color each pixel by its angle bin
    visualization = np.full((H, W), np.nan)
    visualization[freq_mask] = (angles[freq_mask] // bin_size) * bin_size

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization, cmap='hsv', vmin=0, vmax=360)
    plt.colorbar(label='Angle (degrees)')
    plt.axis('off')
    plt.title(f'Angular Bins with {bin_size}° bins and spatial frequency range [{freq_min}-{freq_max}]')
    plt.show()


def mean_power_spectrum_angles(shifted_power_spectrum: np.ndarray, bin_size: int = 10, freq_min: int = 10, freq_max: int = 100) -> np.ndarray:
    H, W = shifted_power_spectrum.shape
    cy, cx = H // 2, W // 2

    x = np.arange(W) - cx
    y = np.arange(H) - cy
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt(xx**2 + yy**2)
    angles = np.degrees(np.arctan2(yy, xx)) % 360

    angular_bins = (angles // bin_size) * bin_size
    unique_bins = np.arange(0, 360, bin_size)

    bins = np.zeros_like(unique_bins, dtype=np.float64)
    bins_counts = np.zeros_like(unique_bins, dtype=np.int64)

    mean_power_by_angle = np.zeros(360 // bin_size)

    for x in range(W):
        for y in range(H):
            #check that the distance from the center is within the specified frequency range
            if freq_min <= dist[y, x] <= freq_max:
                angle_to_center = angles[y, x]
                bin_index = int(angle_to_center // bin_size)
                bins[bin_index] += shifted_power_spectrum[y, x]
                bins_counts[bin_index] += 1
    mean_power_by_angle = bins / np.maximum(bins_counts, 1)  # Avoid division by zero
    return mean_power_by_angle

#task 5 

task5_output = OUTPUT_FOLDER / "task5"
task5_output.mkdir(parents=True, exist_ok=True)

def spatial_derivative(image: np.ndarray, x_order: int, y_order:int) -> np.ndarray:

    H, W = image.shape
    #frequency coordinates
    u = np.fft.fftfreq(W)  
    v = np.fft.fftfreq(H) 

    uu, vv = np.meshgrid(u, v)

    kernel = (2j * np.pi * uu)**x_order * (2j * np.pi * vv)**y_order
    # Apply via multiplication in frequency domain
    F = np.fft.fft2(image)
    result = np.fft.ifft2(F * kernel)

    



    return np.real(result).astype(np.float64)

if __name__ == "__main__":

    #task1
    trui_img = imread(IMAGES_FOLDER / "trui.png") 
    shifted_fourier_transform = fft2_and_shift(trui_img, "trui")

    imsave(task1_output / ("trui_power_shifted.png"), visualize_power_spectrum(shifted_fourier_transform)) 

    #task2 
    cameraman_img = imread(IMAGES_FOLDER / "cameraman.tif")
    imsave(task2_output / "cameraman_original.png", cameraman_img)
    modified_image = add_cos_wave(cameraman_img, "cameraman")
    imsave(task2_output / "cameraman_modified.png", modified_image)

    _ = fft2_and_shift(cameraman_img, "cameraman_original")
    modified_power_spectrum = fft2_and_shift(modified_image, "cameraman_modified")
    _ = filter_planar_waves(modified_image, "cameraman_modified", v0, w0)

    #task 3

    big_ben_image= imread(IMAGES_FOLDER / "bigben_cropped_gray.png")
    sum = radial_average(power_spectrum(big_ben_image))
    log_plot(sum, "big ben radial average", "radius", "mean power", task3_output)

    random_noise = (np.random.rand(big_ben_image.shape[0], big_ben_image.shape[1]) * 255).astype(np.uint8)
    noise_sum = radial_average(power_spectrum(random_noise))
    log_plot(noise_sum, "random noise radial average", "radius", "mean power", task3_output)


    #task 4 
    #visualize_angular_bins()
    angle_values = np.arange(0, 360, 10)
    
    mean_bigben = mean_power_spectrum_angles(power_spectrum(big_ben_image))
    mean_noise = mean_power_spectrum_angles(power_spectrum(random_noise))
    y_max = max(np.max(mean_bigben), np.max(mean_noise))
    y_min = min(np.min(mean_bigben), np.min(mean_noise))

    log_plot(mean_bigben, "big ben mean power by angle", "angle (degrees)", "mean power", task4_output, x_values= angle_values, xlim=(0, 360), ylim=(y_min, y_max))
    log_plot(mean_noise, "random noise mean power by angle", "angle (degrees)", "mean power", task4_output, x_values= angle_values, xlim=(0, 360), ylim=(y_min, y_max))

    
    
    
    #task 5 
    pout_image = imread(IMAGES_FOLDER / "pout.tif")
    result = spatial_derivative(pout_image, x_order=0, y_order=1)

    # For visualization only:
    plt.imshow(result, cmap='bwr')  # bwr is good for signed data (blue-white-red)
    plt.colorbar()                  # the assignment asks for a colorbar!
    plt.savefig(task5_output / "pout_spatial_derivative.png")
    plt.close()





    







