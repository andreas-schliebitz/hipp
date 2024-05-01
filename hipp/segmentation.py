import cv2
import logging
import numpy as np

from typing import Dict
from hipp.utils.utility import imshow
from hipp.background_removal.otsu_thresholding import OtsuThresholding


def calc_average_object_spectrum(
    hypercube: np.ndarray,
    otsu_thresholding: OtsuThresholding,
    min_object_area: int = None,
) -> Dict:
    return {
        k: v.mean(axis=0)
        for k, v in get_object_spectra(
            hypercube, otsu_thresholding, min_object_area
        ).items()
    }


def get_object_spectra(
    hypercube: np.ndarray,
    otsu_thresholding: OtsuThresholding,
    min_object_area: int = None,
    preserve_spatial: bool = False,
    show_contour_mask: bool = False,
) -> Dict:
    hypercube = np.copy(hypercube)
    otsu_thresholding.load(hypercube)
    contour_mask = otsu_thresholding.run()

    if show_contour_mask:
        logging.info(f"Contour mask: {contour_mask.dtype, contour_mask.shape}")
        imshow(contour_mask)

    num_labels, label_im, stats, _ = cv2.connectedComponentsWithStats(contour_mask, 4)

    if min_object_area:
        for obj_num in range(num_labels):
            if stats[obj_num, cv2.CC_STAT_AREA] < min_object_area:
                label_im[label_im == obj_num] = 0

    objects = extract_objects(label_im)
    res = {}
    for comp_num, px_pos in objects.items():
        if preserve_spatial:
            min_y, min_x = px_pos[:, 0].min(), px_pos[:, 1].min()
            max_y, max_x = px_pos[:, 0].max(), px_pos[:, 1].max()

            # Create blank hypercube for object's hyperpixels
            vals = np.zeros((max_y - min_y + 1, max_x - min_x + 1, hypercube.shape[-1]))

            # Project global pixel positions onto origin (top left) of vals
            vals[px_pos[:, 0] - min_y, px_pos[:, 1] - min_x, :] = hypercube[
                px_pos[:, 0], px_pos[:, 1], :
            ]
        else:
            vals = hypercube[px_pos[:, 0], px_pos[:, 1], :]

        res[comp_num] = vals
    return res


def extract_objects(labels_im: np.ndarray) -> Dict:
    objects = {}
    for comp_num in np.unique(labels_im):
        if comp_num == 0:
            continue
        objects[comp_num] = np.argwhere(labels_im == comp_num)
    return objects
