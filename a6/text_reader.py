import numpy as np 
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rotate
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import closing, disk
from const import IMAGES_FOLDER, OUTPUT_FOLDER

def automatic_rotation(image:NDArray) -> NDArray: 
    """
    Detects corners in the input image and rotates it to align with the detected features.
    """
    #Find opposing sides of the label in the image by detecting corners


    #Preprocess the image by thresholding to enhance corner detection 



    filled_image = closing(image, disk(30))

    image_th = np.where(filled_image > 140, 255, 0).astype(np.uint8)

    harris_response = corner_harris(image_th, sigma= 55, k=0.1)
    corners = corner_peaks(
        harris_response,
        num_peaks=2, min_distance=70
    )


    savepath = OUTPUT_FOLDER / "task2" / "corners_detected.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(image_th, cmap='gray')
    plt.plot(corners[:, 1], corners[:, 0], 'r+', markersize=6, markeredgewidth=1.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath)
    #Calculate the angle of rotation based on the detected points stretching out the horizontal line
    
    edge_vector = corners[1] - corners[0]
    #calculate the angle between the edge vector and the horizontal axis

    angle = np.degrees(np.arctan2(edge_vector[0], edge_vector[1]))

    print(f"Angle of rotation: {angle:.2f} degrees")

    #Rotate the image by the calculated angle to align it with the horizontal axis

    rotated_image = rotate(image, angle-90, preserve_range=True, resize=True).astype(np.uint8)
    

    #rotate a further 90 degrees to align the text in the correct orientation
    rotated_image = np.where(rotated_image > 140, 255, 0).astype(np.uint8)

    return rotated_image

if __name__ == "__main__":
    output = OUTPUT_FOLDER / "task2"
    output.mkdir(exist_ok=True, parents=True)
    image = io.imread(IMAGES_FOLDER / "textlabel_gray_small.png")
    rotated_image = automatic_rotation(image)
    io.imsave(output / "textlabel_rotated_th.png", rotated_image.astype(np.uint8))
