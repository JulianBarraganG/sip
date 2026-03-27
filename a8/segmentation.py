from typing import Any
import numpy as np
from numpy.typing import NDArray

from a8.plotting import plot_true_vs_pred_seg


def extract_patches(
    image: NDArray[np.uint16], k_size: int = 29
) -> NDArray[np.float32]:
    W, H = image.shape
    assert k_size % 2 == 1, "Kernel size must be odd i.e. have a single center."
    pad = k_size // 2
    padded_img = np.pad(
        image, pad, mode='constant', constant_values=0
    ).astype(np.float32)
    
    # Shape (65536, 29, 29, 1) to match the model's expected input
    patches = np.zeros((W * H, k_size, k_size, 1), dtype=np.float32)
    
    k = 0
    for i in range(H):
        for j in range(W):
            patch = padded_img[i : i + k_size, j : j + k_size] / 255.
            patches[k, :, :, 0] = patch
            k += 1

    return patches

def dice_score(
    y_true: NDArray[np.float32],
    y_pred: NDArray[np.float32],
    num_classes: int = 135,
) -> list[Any]:
    """
    Computes the mean Dice coefficient across all classes present in the images.
    """
    dice_scores = []
    
    # We iterate through every possible class index
    for i in range(num_classes):
        # Create binary masks for the current class
        true_class = (y_true == i)
        pred_class = (y_pred == i)
        
        # Calculate intersection and sums
        intersection = (true_class * pred_class).sum()
        total_pixels = true_class.sum() + pred_class.sum()
        
        # Only compute dice if the class exists in either the truth or the prediction
        if total_pixels > 0:
            dice = (2. * intersection) / total_pixels
            dice_scores.append(dice)
            
    return dice_scores

if __name__ == "__main__":
    from a8.const import (
        TEST_IMAGES_FOLDER,
        TEST_SEGMENTATIONS_FOLDER,
        OUTPUT_FOLDER,
    )
    from a8.plotting import plot_original_vs_seg
    from skimage.io import imread

    KERNEL_SIZE = 29
    OUT = OUTPUT_FOLDER / "task3"
    OUT.mkdir(parents=False, exist_ok=True)

    test_images_folder = TEST_IMAGES_FOLDER

    test_img = imread(test_images_folder / "1128_3_image.png") # shape (256, 256)
    patches = extract_patches(test_img, KERNEL_SIZE)

    # Load the model and classify each pixel.
    from a8.keras_model import model
    preds = model.predict(patches, batch_size=256) # shape (65536, 135)
    # Make segmentation mask
    am = preds.argmax(axis=1)
    num_unique = len(np.unique(am))
    seg_mask = am.reshape(test_img.shape)

    # Plot side-by-side
    title = (
        "Original brain scan vs classified labels."
        f"\n{num_unique}/135 Classes represented with distinct colors."
    )
    save_path = OUT / "1128_vs_predicted_seg.png"
    plot_original_vs_seg(test_img, seg_mask, title, save_path)


    ### Task 3.4
    true_seg_path = TEST_SEGMENTATIONS_FOLDER / "1128_3_seg.png"
    true_seg = imread(true_seg_path)

    # Flatten or ensure they are the same shape for comparison
    # true_seg is (256, 256), seg_mask is (256, 256)
    dice_scores = dice_score(true_seg, seg_mask)
    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")

    # Optional: Plot the ground truth correctly to see if it matches your prediction
    tvp_save_path = OUT / "true_vs_pred_mask.png"
    plot_true_vs_pred_seg(true_seg, seg_mask, tvp_save_path)
