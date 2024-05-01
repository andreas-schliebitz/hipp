import numpy as np
from hipp.core.module import Module


class MultiplicativeScatterCorrection(Module):
    def __init__(self, reference_spectrum: np.ndarray = None) -> None:
        self.reference_spectrum = reference_spectrum
        super().__init__(name=__name__, path=__file__, normalize_output=False)

    def _mean_centre_correction(self) -> None:
        pixel_means = self.data.mean(axis=-1)[:, :, np.newaxis]
        self.data -= pixel_means

    def run(self) -> np.ndarray:
        self._mean_centre_correction()
        self._msc()
        return self.reference_spectrum

    def _msc(self):
        if self.reference_spectrum is None:
            self.reference_spectrum = self.data.mean(axis=(0, 1))

        self.data = np.apply_along_axis(self._correct, -1, self.data)

    def _correct(self, pixel) -> np.ndarray:
        if not pixel.count():
            return pixel

        coef = np.polynomial.Polynomial.fit(
            self.reference_spectrum, pixel, deg=1, full=False
        ).coef

        return (pixel - coef[0]) / coef[1]

    def __str__(self) -> str:
        return f"""MultiplicativeScatterCorrection(
            reference_spectrum = {self.reference_spectrum}
        )"""
