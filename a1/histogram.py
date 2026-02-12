from email.mime import image
import os
import numpy as np 
from skimage.io import imread, imshow, show, imsave
from skimage import util
from const import IMAGES_FOLDER, OUTPUT_FOLDER
import matplotlib.pyplot as plt

#task 1 
pout_filepath = os.path.join(IMAGES_FOLDER, "pout.tif")
pout = imread(pout_filepath)

def histogram_uint8(image: np.ndarray) -> np.ndarray:
    """Given an image as a (N, M) numpy array, computes the histogram of pixel values.
    The output is a (256,) numpy array, where the i-th element is the number of pixels
    in the input image with value i."""
    values, counts = np.unique(image, return_counts=True)

    histogram = np.zeros(256, dtype=np.int64)
    histogram[values] = counts

    return histogram

def plot_uint8_histogram(histogram: np.ndarray, title: str) -> None:
    """Given a histogram as a (256,) numpy array, plots the histogram as a bar plot."""
    savepath = os.path.join(OUTPUT_FOLDER, "Histogram_" + title + ".png")
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(256), histogram)
    plt.xlabel("Pixel value")
    plt.ylabel("Number of pixels")
    plt.title("Histogram of pixel values for image: " + title)
    plt.savefig(savepath)
    plt.close()


#Just used for visualization, not required for the assignment
def cdf_from_histogram(histogram: np.ndarray) -> np.ndarray:
    """
    Given a histogram as a numpy array, computes the cdf as a numpy array
    Normalizes to [0,1] by default
    """
    cdf = np.cumsum(histogram)
    
    cdf_normalized = cdf / np.max(cdf) #normalize to [0, 1] 
    return cdf_normalized
    

def plot_uint8_cdf(cdf: np.ndarray, title: str) -> None:
    """Given a cumulative distribution function as a (256,) numpy array, plots the CDF as a line plot."""
    savepath = os.path.join(OUTPUT_FOLDER, "CDF_" + title + ".png")
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(256), cdf)
    plt.xlabel("Pixel value")
    plt.ylabel("Cumulative number of pixels")
    plt.title("Cumulative distribution function of pixel values for image: " + title)
    plt.savefig(savepath)
    plt.close()

plot_uint8_histogram(histogram_uint8(pout), "pout")
plot_uint8_cdf(cdf_from_histogram(histogram_uint8(pout)), "pout")


#task 2 

def cdf_to_intensity(image: np.ndarray, cdf: np.ndarray, title: str) -> np.ndarray:
    """
    Given an image and its CDF, computes the floating-point image C(I) 
    such that the intensity at each pixel(x, y) is C(I(x, y)). 
    This image is saved to output as a .png file.
    """
    savepath = os.path.join(OUTPUT_FOLDER, "intensity_" + title + ".png")
    intensity_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity_image[i, j] = cdf[image[i, j]]
    intensity_image_uint8 = util.img_as_ubyte(intensity_image)
    imsave(savepath, intensity_image_uint8)
    return intensity_image_uint8

intensity_img = cdf_to_intensity(pout, cdf_from_histogram(histogram_uint8(pout)), "pout")

#These plots do a good job of showing that the intensity image has a more uniform distribution of pixel values, as the histogram is more flat and the CDF is more linear.

plot_uint8_histogram(histogram_uint8(intensity_img), "pout_intensity")
plot_uint8_cdf(cdf_from_histogram(histogram_uint8(intensity_img)), "pout_intensity")

#NOTeE FOR REPORT: We now use the full dynamic range 

#task 3 

def pseudo_inverse_cdf_uint8(cdf: np.ndarray, l: float) -> np.ndarray:
    """
    Given a cumulative distribution function as a numpy array, and some value l in [0, 1],
    computes the pseudo-inverse CDF, i.e., finds the smallest intensity i such that C(i) >= l.
    """

    return np.min(np.where(cdf >= l))

#task 4 

def histogram_matching_uint8(source_image: np.ndarray, reference_image: np.ndarray, title: str) -> np.ndarray:
    """
    Given a source Image I1 and a reference Image I2 as numpy arrays, performs histogram matching
    and returns + saves the matched image to output as a .png file.
    """

    savepath = os.path.join(OUTPUT_FOLDER, "matched_" + title.replace(" ", "_") + ".png")

    source_histogram = histogram_uint8(source_image) 
    reference_histogram = histogram_uint8(reference_image) 

    source_cdf = cdf_from_histogram(source_histogram)
    reference_cdf = cdf_from_histogram(reference_histogram)

    resulting_image = np.zeros_like(source_image, dtype=np.uint8)

    

    for i in range(source_image.shape[0]):
        for j in range(source_image.shape[1]):
            resulting_image[i,j] = pseudo_inverse_cdf_uint8(reference_cdf, source_cdf[source_image[i, j]])

    imsave(savepath, resulting_image)
    return resulting_image

spine_filepath = os.path.join(IMAGES_FOLDER, "spine.tif")
spine = imread(spine_filepath)
matched_img = histogram_matching_uint8(pout, spine, "pout to spine")

matched_histogram = histogram_uint8(matched_img)
spine_histogram = histogram_uint8(spine)

matched_cdf = cdf_from_histogram(matched_histogram)
spine_cdf = cdf_from_histogram(spine_histogram)

plot_uint8_histogram(matched_histogram, "matched_image")
plot_uint8_histogram(spine_histogram, "spine_image")

plot_uint8_cdf(matched_cdf, "matched_image")
plot_uint8_cdf(spine_cdf, "spine_image")


    




    
    

    

