import numpy.ma as ma
from hipp.core.module import Module
from scipy.signal import savgol_filter


class SavitzkyGolayFilter(Module):
    def __init__(
        self,
        window_width: int = None,
        polynomial_order: int = 2,
        prior_nth_derivative: int = 0,
    ) -> None:
        self.window_width = window_width
        self.polynomial_order = polynomial_order
        self.prior_nth_derivative = prior_nth_derivative
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        if self.window_width is None:
            ww = self.data.shape[-1] // 4
            if ww % 2 == 0:
                w += 1
            self.window_width = ww

        self.data = ma.masked_array(
            savgol_filter(
                self.data,
                window_length=self.window_width,
                polyorder=self.polynomial_order,
                deriv=self.prior_nth_derivative,
                axis=-1,
            ),
            mask=self.data.mask,
            fill_value=0,
        )

    def __str__(self) -> str:
        return f"""SavitzkyGolayFilter(
            window_width = {self.window_width}
            polynomial_order = {self.polynomial_order}
            prior_nth_derivative = {self.prior_nth_derivative}
        )"""
