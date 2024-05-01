import numpy as np
from hipp.core.module import Module
from hipp.utils.utility import interpolate, to_cube

from numba import njit
from scipy.stats import skew
from statsmodels.distributions.empirical_distribution import ECDF

"""
    Distilled version of PyOD copod.py implementation.
    See: https://github.com/yzhao062/pyod/blob/master/pyod/models/copod.py
"""


class CopulaBasedOutlierDetection(Module):
    def __init__(self, contamination: float = 0.05) -> None:
        assert 0 < contamination < 0.5

        self.contamination = contamination
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        X = self.data[~self.data.mask.any(axis=-1)]
        self.data = to_cube(self._copod(X))

    def _copod(self, X: np.ndarray) -> None:
        U = np.apply_along_axis(self._ecdf, 0, X)
        V = np.apply_along_axis(self._ecdf, 0, -X)

        U, V = _negative_log_transform(U, V)

        b = skew(X, axis=0)
        W = _unskew(b, X, U, V)

        O_score = _score(U, V, W)

        return self._interpolate(X, O_score)

    def _interpolate(self, X: np.ndarray, O_score: np.ndarray) -> None:
        n = O_score.shape[0]
        count = int(np.floor(n * self.contamination))

        if not count:
            return X

        where = np.zeros(n)
        top = np.argpartition(O_score, -count)[-count:]
        where[top] = True
        return interpolate(X, where)

    @staticmethod
    def _ecdf(X):
        ecdf = ECDF(X)
        return ecdf(X)

    def __str__(self) -> str:
        return f"""CopulaBasedOutlierDetection(
            contamination = {self.contamination}
        )"""


@njit(cache=True)
def _negative_log_transform(U, V):
    return -1 * np.log(U), -1 * np.log(V)


@njit(cache=True)
def _unskew(b, X, U, V):
    s = np.sign(b)
    return U * -1 * np.sign(s - 1) + V * np.sign(s + 1)


@njit(cache=True)
def _score(U, V, W):
    return np.maximum(W, np.add(U, V) / 2).sum(axis=1)
