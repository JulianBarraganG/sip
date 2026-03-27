import numpy as np
from a6.transform import white_square  
from const import OUTPUT_FOLDER 
from numpy.typing import NDArray 
from skimage.transform import warp  
import matplotlib.pyplot as plt

def TRS(image: NDArray, t: tuple, theta: float, s: float): 
    assert s >=0, f"s must be greater than 0, got: {s}"

    assert len(t) == 2, f"t must have a length of exactly 2, got {len(t)}"
    
    assert 0 <= theta <= 2*np.pi, f"theta must be in range [0, 2π], got {theta}" 

    Tt_inv = np.identity(3)
    Tc = np.identity(3)
    Tc_inv = np.identity(3)
    S_inv = np.identity(3)
    R_inv = np.identity(3)

    for i in range(2): 
        Tt_inv[i,2] = -1*t[i] 
        Tc[i,2] = image.shape[i] // 2
        Tc_inv[i,2] = -1*(image.shape[i]//2) 

    S_inv[0,0] = 1/s 
    S_inv[1,1] = 1/s 

    R_inv[0,0] = np.cos(-theta) 
    R_inv[1,0] = -1*np.sin(-theta) 
    R_inv[0,1] = np.sin(-theta) 
    R_inv[1,1] = np.cos(-theta) 

    inv_matrix = Tc @ S_inv @ R_inv @ Tc_inv @ Tt_inv 

    transformed_img = warp(image, inv_matrix, order=0, mode='constant', cval = 0)
    
    return transformed_img.astype(np.uint8) 
    
    
if __name__ == "__main__": 
    out_path = OUTPUT_FOLDER / "task1"  
    out_path.mkdir(parents=False, exist_ok=True )
    square = white_square(101) 
    transformed_square  = TRS(image= square, t= (10.4, 15.7), theta= np.pi/10, s = 2) 
    plt.imshow(transformed_square, cmap = 'gray')
    plt.savefig(out_path / "transformed_square.png") 
    plt.close()

