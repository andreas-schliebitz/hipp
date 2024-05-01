import numpy as np
import numpy.ma as ma
from numba import njit

from hipp.core.module import Module
from hipp.utils.utility import interpolate, invertible_cov, is_masked_cube

"""
    Optimized and improved version of a MATLAB implementation.
    See: https://github.com/fverdoja/LAD-Laplacian-Anomaly-Detector/blob/master/rxd.m
"""


class ReedXiaoliDetector(Module):
    def __init__(self, confidence_coeff: float = 0.98) -> None:
        assert 0.5 < confidence_coeff < 1

        self.confidence_coeff = confidence_coeff
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        rx_scores = self._rxd()
        outlier_coords, rx_threshold, cdf = self._select_outliers(rx_scores)

        for y, x in outlier_coords:
            where = np.zeros(rx_scores.shape[1], dtype=bool)
            where[x] = True
            self.data[y, :, :] = np.apply_along_axis(
                interpolate, 0, self.data[y, :, :], where
            )

        assert is_masked_cube(self.data)
        return rx_threshold, cdf

    def _rxd(self) -> np.ndarray:
        X = ma.masked_array.copy(self.data)

        h, w, d = X.shape
        X = X.reshape((h * w, d))

        M = X.mean(axis=0)
        C = invertible_cov(X)
        Q = np.linalg.inv(C)

        return np.apply_along_axis(_calc_rx_scores, 1, X, M, Q).reshape(h, w)

    def _select_outliers(self, rx_scores: np.ndarray) -> np.ndarray:
        rx_scores = ((rx_scores / rx_scores.max()) * 255).astype(np.uint8)

        uniques, count = np.unique(rx_scores, return_counts=True)
        pdf = count / np.prod(rx_scores.shape)
        cdf = np.cumsum(pdf)

        rx_threshold = uniques[cdf > self.confidence_coeff][0]
        return np.argwhere(rx_scores > rx_threshold), rx_threshold, cdf

    def __str__(self) -> str:
        return f"""ReedXiaoliDetector(
            confidence_coeff = {self.confidence_coeff}
        )"""


@njit(cache=True)
def _calc_rx_scores(X: np.ndarray, M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if not X.any():
        return 0
    xM = (X - M).astype(Q.dtype)
    return xM @ Q @ xM.T
