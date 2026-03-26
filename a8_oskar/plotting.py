from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np 
from numpy.typing import NDArray 
import pandas as pd 
from typing import Any 
def closed_curves(data: NDArray[Any], savepath: Path):
    num_obj = len(data) 
    colors = plt.get_cmap("tab10") 
    plt.figure(figsize=(10,6))
    for i in range(num_obj): 
        
        plt.plot(data[i, 0] , data[i, 1], color=colors(i), marker='.', label=f"Wing number {i}") 
    plt.legend()
    plt.savefig(savepath)
    plt.close()
