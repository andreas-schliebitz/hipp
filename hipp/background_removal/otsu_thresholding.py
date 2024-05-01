import cv2
import numpy as np

from hipp.background_removal.common import (
    contours_to_mask,
    find_contours,
    remove_background,
)
from hipp.core.module import Module
from hipp.utils.utility import normalize, is_normalized


class OtsuThresholding(Module):
    def __init__(self, contrast_gain: float = 1.5) -> None:
        assert 1.0 <= contrast_gain <= 3.0

        self.contrast_gain = contrast_gain
        super().__init__(name=__name__, path=__file__)

    def run(self) -> np.ndarray:
        max_contrast_band = self.max_rms_contrast()

        if not is_normalized(max_contrast_band):
            max_contrast_band = normalize(max_contrast_band)

        src = cv2.convertScaleAbs(
            (max_contrast_band * 255).astype(np.uint8),
            alpha=self.contrast_gain,
            beta=1.0,
        )

        contour_mask = normalize(
            cv2.threshold(
                src,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1]
        )

        contours = find_contours(contour_mask)
        contour_mask = contours_to_mask(contours, contour_mask.shape)
        remove_background(self.data, contour_mask)

        return contour_mask

    def max_rms_contrast(self) -> np.ndarray:
        return self.data[..., np.argmax(self.data.std(axis=(0, 1)))]

    def __str__(self) -> str:
        return f"""OtsuThresholding(
            contrast_gain = {self.contrast_gain}
        )"""
