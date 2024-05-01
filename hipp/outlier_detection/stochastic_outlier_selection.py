import numpy as np
import numpy.ma as ma

from numba import njit
from hipp.core.module import Module
from hipp.utils.utility import interpolate, to_cube, is_masked_cube


"""
    Stochastic Outlier Selection by Jeroen Janssens, 2013
    See: https://datascienceworkshops.com/blog/stochastic-outlier-selection/
         https://github.com/jeroenjanssens/scikit-sos/blob/master/sksos/sos.py
"""


class StochasticOutlierSelection(Module):
    def __init__(
        self,
        outlier_probability_threshold: float = 0.51,
        perplexity: float = 30,
        max_tries: int = 5000,
        eps: float = 1e-5,
        collapse_samples: bool = True,
    ) -> None:
        assert 0.5 < outlier_probability_threshold < 1

        self.outlier_probability_threshold = outlier_probability_threshold
        self.perplexity = perplexity
        self.max_tries = max_tries
        self.eps = eps
        self.collapse_samples = collapse_samples
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        X = self.data[~self.data.mask.any(axis=-1)]

        if self.collapse_samples:
            X = X.mean(axis=0).T[..., np.newaxis]

        O = self._sos(X)
        where = O > self.outlier_probability_threshold

        if self.collapse_samples:
            self.data = ma.masked_array(
                interpolate(self.data.transpose(-1, 0, 1), where).transpose(1, -1, 0),
                mask=self.data.mask,
                fill_value=0,
            )
        else:
            self.data = to_cube(interpolate(X, where))

        assert is_masked_cube(self.data)

    def _sos(self, X: np.ndarray) -> np.ndarray:
        D = _x2d(X)
        A = self._d2a(D)
        B = self._a2b(A)
        O = self._b2o(B)
        return O

    def _affinity(self, i: int, D: np.ndarray, beta: np.ndarray):
        betamin, betamax = -np.inf, np.inf
        n = D.shape[0]

        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        H, thisA = _get_perplexity(Di, beta[i])
        logU = np.log(self.perplexity)

        Hdiff = H - logU
        tries = 0
        while (np.isnan(Hdiff) or np.abs(Hdiff) > self.eps) and tries < self.max_tries:
            if np.isnan(Hdiff):
                beta[i] = beta[i] / 10.0
            elif Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0
            H, thisA = _get_perplexity(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        return thisA

    # Computes affinity matrix A from dissimilarity matrix D.
    def _d2a(self, D: np.ndarray) -> np.ndarray:
        n = D.shape[0]
        A = np.zeros((n, n))
        beta = np.ones((n, 1))

        for i in range(n):
            thisA = self._affinity(i, D, beta)
            A[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisA

        return A

    @staticmethod
    def _a2b(A: np.ndarray) -> np.ndarray:
        return A / A.sum(axis=1)[:, np.newaxis]

    @staticmethod
    def _b2o(B: np.ndarray) -> np.ndarray:
        return np.prod(1 - B, 0)

    def __str__(self) -> str:
        return f"""StochasticOutlierSelection(
            outlier_probability_threshold = {self.outlier_probability_threshold}
            perplexity = {self.perplexity}
            max_tries = {self.max_tries}
            eps = {self.eps}
        )"""


# Computes dissimilarity matrix D from input matrix X.
@njit(cache=True)
def _x2d(X: np.ndarray) -> np.ndarray:
    sumX = np.sum(np.square(X), 1)
    return np.sqrt(np.abs(np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX)))


@njit(cache=True)
def _get_perplexity(D: np.ndarray, beta: float) -> tuple:
    A = np.exp(-D * beta)
    sumA = np.sum(A)
    H = np.log(sumA) + beta * np.sum(D * A) / sumA
    return H, A
