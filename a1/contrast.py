import numpy as np 

from matplotlib.pyplot import imshow, show 
from skimage.io import imread

def gamma_transform(image: np.ndarray, gamma:float):

    transformed_img = np.pow(image,gamma) 

    c = np.max(image)/ np.max(transformed_img)
    

    return c * transformed_img 


test_img = imread('../data/week1/cameraman.tif') 

imshow(test_img, cmap = 'gray')



gamma_img = gamma_transform(test_img, gamma = 0.2) 

imshow(gamma_img, cmap = 'gray') 

show()
