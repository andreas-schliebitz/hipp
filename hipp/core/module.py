import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import os
import numpy as np
import numpy.ma as ma

from hipp.utils.utility import is_masked_cube, is_normalized, normalize


class Module:
    def __init__(self, name: str, path: str, normalize_output: bool = True) -> None:
        self.name = name
        self.data = None
        self.path = os.path.dirname(os.path.realpath(path))
        self.normalize_output = normalize_output

        logging.info(f"Created {self}")

    def load(self, data: np.ndarray, mask: np.ndarray = None) -> None:
        assert data is not None and len(data.shape) == 3

        if not is_masked_cube(data):
            assert data.min() >= 0

            if mask is None:
                mask = data.sum(axis=-1).astype(bool)

            assert mask.dtype == bool and len(mask.shape) == 2 and (mask == True).any()

            data = ma.masked_array(
                data,
                mask=~np.repeat(mask, data.shape[-1], axis=-1),
                fill_value=0,
            )

        self.data = data

        assert is_masked_cube(self.data) and not np.isnan(self.data).any()

    def get_data(self) -> np.ndarray:
        assert self.data is not None and len(self.data.shape) == 3

        if np.iscomplexobj(self.data):
            self.data = np.real(self.data)

        if self.normalize_output and not is_normalized(self.data):
            self.data = ma.masked_array(
                normalize(self.data), mask=self.data.mask, fill_value=0
            )

        assert is_masked_cube(self.data) and not np.isnan(self.data).any()
        return self.data

    def __str__(self) -> str:
        return f"""Module(
            name={self.name}
            data=(shape={self.data.shape}, dtype={self.data.dtype})
            path={self.path})
            normalize_output={self.normalize_output}
        )"""
