import cv2
import numpy as np


def remove_background(data: np.ndarray, contour_mask: np.ndarray) -> None:
    for b in range(data.shape[-1]):
        data[..., b] = np.where(contour_mask == 1, data[..., b], 0)


def find_contours(contour_mask: np.ndarray) -> np.ndarray:
    return cv2.findContours(
        (contour_mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS,
    )[0]


def contours_to_mask(contours: np.ndarray, mask_shape: tuple) -> np.ndarray:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, contours, 1)
    return mask
