import numpy as np 

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv, hsv2rgb

def single_channel_gamma_transform_uint8(
    image: np.ndarray,
    gamma: float
) -> np.ndarray:
    """Given an image as a (N, M) numpy array, makes gamma correction 
    transformation"""
    transformed_img = np.pow(image.astype(np.float64), gamma)
    c = np.pow(255., 1-gamma) # assumes uint8 discretization

    return np.round(c * transformed_img).astype(np.uint8)

def gamma_transform_uint8(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:
    try:
        num_channels = image.shape[2]
        scaled_channels = [
            single_channel_gamma_transform_uint8(image[:,:,i], gamma) for i in range(num_channels)
        ]
        return np.stack(scaled_channels, axis=2)
    except IndexError:
        assert len(image.shape) == 2, "immage must be 2D"
        return single_channel_gamma_transform_uint8(image, gamma)

def gamma_transform_value_uint8(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Given an RGB image as a (N, M, 3) numpy array, makes gamma correction"""
    assert len(image.shape) == 3, "assumes RGB image input"
    hsv_img = rgb2hsv(image) # scales to [0,1] interval
    hsv_img = (255*hsv_img).astype(np.uint8) # rescale to uint8
    gc_val_channel = single_channel_gamma_transform_uint8(hsv_img[:,:,2], gamma)
    hsv_img[:,:,2]= gc_val_channel


    return hsv2rgb(hsv_img.astype(np.uint8))


if __name__ == "__main__":
    from const import IMAGES_FOLDER

    test_img = imread(IMAGES_FOLDER / "autumn.tif") 
    transformed = gamma_transform_value_uint8(test_img, 1)
