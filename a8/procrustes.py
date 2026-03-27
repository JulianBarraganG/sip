from typing import Any
from numpy._typing import NDArray
import numpy as np

def get_rotation(target: NDArray[Any], other: NDArray[Any]) -> NDArray[Any]:
    """
    Get the optimal rotation matrix R that aligns `other` to `target` using SVD.
    Parameters
    ----------
    target
        The target shape (2, N) to whoms orientation we want to align.
    other
        The shape (2, N) that we want to rotate to align with the target.
    Returns
    -------
    NDArray[Any]
        The optimal rotation matrix R (2, 2) that aligns `other` to `target`.
    """
    A = other @ target.T
    U, _, Vt = np.linalg.svd(A)
    V = Vt.T
    return V @ U.T


def procrustes(S: NDArray[Any], target: NDArray[Any]) -> NDArray[Any]:
    """
    Note
    ----
        Assumes number of dims 2 i.e. (x,y)
    """

    M, d, N = S.shape
    assert d == 2, "Only 2D coordinates are supported."
    assert target.shape == (2, N), (
        f"Target shape must be (2, N)={(2, N)}, but got {target.shape}."
    )

    ### Translate all coordinates to origo
    x_center = S[:, 0, :].mean(axis=1, keepdims=True)  # shape (182, 1)
    y_center = S[:, 1, :].mean(axis=1, keepdims=True)  # shape (182, 1)

    centered = S.copy()
    centered[:, 0, :] -= x_center  # broadcasts across 32 landmarks
    centered[:, 1, :] -= y_center
    target -= target[0].mean(axis=0)  # Center the target as well
    target -= target[1].mean(axis=0)  # Center the target as well

    ### Scale all of them w.r.t. target
    # for all N sum inner prod of y-vector and x-vectors
    # for all N sum inner prod of y-vector with itself i.e. norm squared
    # numerator: sum(target * centered[i]) for each i -> shape (M,)
    enum = np.sum(centered * target, axis=(1, 2))
    denom = np.sum(centered * centered, axis=(1, 2))
    # Check for 0 division
    if np.any(denom == 0):
        print(
            "WARNING: Some shapes have zero variance. "
            "To avoid division by zero, 0-s are set to 1.0"
        )
        denom = np.where(denom == 0, 1.0, denom)  # avoid division by zero
    s = enum / denom
    output = centered * s[:, np.newaxis, np.newaxis]  # broadcast over (2, 32)

    ### Rotate them all w.r.t. target
    # Take SVD of YX^T and get USV
    # Let R = V @ U^T
    # Rotate every wing except target by R
    for i in range(M):
        other = output[i]
        R = get_rotation(target, other)
        output[i] = R @ other

    return output
