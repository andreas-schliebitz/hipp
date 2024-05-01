import numpy as np

from numba import njit
from hipp.core.module import Module
from hipp.utils.utility import to_cube, is_masked_cube


class SpatialBinning(Module):
    def __init__(
        self,
        reference_spectrum: np.ndarray = None,
        window_size: tuple = (3, 3),
        binning_perc: float = 0.1,
    ) -> None:
        assert 0 <= binning_perc <= 1

        self.reference_spectrum = reference_spectrum
        self.window_size = window_size
        self.binning_perc = binning_perc

        super().__init__(name=__name__, path=__file__, normalize_output=False)

    def run(self) -> None:
        if is_masked_cube(self.data):
            self.data = self.data.data

        if self.reference_spectrum is None:
            self.reference_spectrum = self.data.mean(axis=(0, 1))

        h, w, d = self.data.shape
        window_h, window_w = self.window_size

        h_steps, h_rem = np.divmod(h, window_h)
        w_steps, w_rem = np.divmod(w, window_w)

        if h_rem != 0:
            h_steps += 1
        if w_rem != 0:
            w_steps += 1

        tile_scores = np.zeros((h_steps, w_steps, 5))
        for h_step in range(h_steps):
            for w_step in range(w_steps):
                tile_scores[h_step, w_step, :] = self._bin(h_step, w_step)

        bin_tile_scores = _get_binning_tile_scores(tile_scores, self.binning_perc)

        for y, x, th, tw, _ in bin_tile_scores:
            y, x, th, tw = int(y), int(x), int(th), int(tw)
            tile = self.data[y : y + th, x : x + tw, :]
            mask = np.full((th, tw, d), -1.0)
            mask[0, 0, :] = tile.mean(axis=(0, 1))
            self.data[y : y + th, x : x + tw, :] = mask

        self.data = to_cube(self.data[self.data[..., 0] != -1, :])

        assert is_masked_cube(self.data)

    def _bin(self, h_step: int, w_step: int) -> list:
        window_h, window_w = self.window_size
        y = h_step * window_h
        x = w_step * window_w
        tile = self.data[y : y + window_h, x : x + window_w, :]

        tile_scores = np.apply_along_axis(_sidsam, -1, tile, self.reference_spectrum)
        th, tw = tile_scores.shape
        return [y, x, th, tw, tile_scores.mean()]

    def __str__(self) -> str:
        return f"""SpatialBinning(
            window_size = {self.window_size},
            binning_perc = {self.binning_perc}
        )"""


@njit(cache=True)
def _get_binning_tile_scores(
    tile_scores: np.ndarray, binning_perc: float
) -> np.ndarray:
    th, tw, td = tile_scores.shape
    tile_scores = np.reshape(tile_scores, (th * tw, td))
    tile_scores = tile_scores[tile_scores[:, -1].argsort()]

    k = int(np.floor(binning_perc * len(tile_scores)))
    return tile_scores[:k]


@njit(cache=True)
def _sidsam(T: np.ndarray, R: np.ndarray) -> float:
    if not T.any():
        return np.inf
    alpha = np.arccos(np.sum(T * R) / (np.sqrt(np.sum(T**2)) * np.sqrt(np.sum(R**2))))
    return _sid(T, R) * np.tan(alpha)


@njit(cache=True)
def _sid(T: np.ndarray, R: np.ndarray) -> float:
    lhs, rhs = 0, 0
    T_sum = np.sum(T)
    R_sum = np.sum(R)
    eps = 2.220446049250313e-16
    for i, _ in enumerate(R):
        T_i = T[i] if T[i] != 0 else eps
        R_i = R[i] if R[i] != 0 else eps
        p_i = T_i / T_sum
        q_i = R_i / R_sum
        lhs += p_i * np.log(p_i / q_i)
        rhs += q_i * np.log(q_i / p_i)
    return lhs + rhs
