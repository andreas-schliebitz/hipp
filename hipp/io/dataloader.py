import numpy as np
import spectral.io.envi as envi

from hipp.utils.utility import path
from typing import Dict


def _get_color_depth(envi_header: Dict):
    if "color depth" in envi_header:
        color_depth = int(envi_header["color depth"])
    else:
        color_depth = 1
    return color_depth


def load_envi(envi_header_filepath: str, color_depth: int = 0) -> np.ndarray:
    envi_header_filepath = path(envi_header_filepath)
    envi_header = envi.read_envi_header(envi_header_filepath)

    if color_depth == 0:
        color_depth = _get_color_depth(envi_header)

    return envi.open(envi_header_filepath).load() / (2**color_depth - 1), envi_header


def save_envi(
    hypercube: np.ndarray,
    envi_header_output_filepath: str,
    envi_header: Dict,
    color_depth: int = 0,
    force=True,
    copy=False,
) -> None:
    if copy:
        hypercube = np.copy(hypercube)

    if color_depth == 0:
        color_depth = _get_color_depth(envi_header)

    hypercube *= (2**color_depth) - 1

    np_dtype = np.uint8
    if 8 < color_depth <= 16:
        np_dtype = np.uint16
    elif color_depth == 32:
        np_dtype = np.uint32

    envi.save_image(
        envi_header_output_filepath,
        hypercube,
        metadata=envi_header,
        dtype=np_dtype,
        force=force,
    )
