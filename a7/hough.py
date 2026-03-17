import numpy as np 
from skimage.feature import canny
from numpy.typing import NDArray

def hough_transform(edges:NDArray): 
    d = np.round(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    acc = np.zeros((2*d, 180))

    for x in range(edges.shape[0]): 
        for y in range(edges.shape[1]): 
            if edges[x,y] == 0: 
                pass
            else: 
                for theta in range(181):
                    rho = np.round(x * np.cos(theta) + y * np.sin(theta) )
                    acc[rho + d, theta] += 1  


