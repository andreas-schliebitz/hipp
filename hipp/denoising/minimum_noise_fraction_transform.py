#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import spectral as spy
from hipp.core.module import Module
from hipp.utils.utility import invertible_cov, is_masked_cube


class MinimumNoiseFractionTransform(Module):
    def __init__(self, denoise_snr=10) -> None:
        self.denoise_snr = denoise_snr
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        h, w, d = self.data.shape
        signal = spy.calc_stats(self.data)
        signal.cov = invertible_cov(np.reshape(self.data, (h * w, d)))

        noise = spy.noise_from_diffs(self.data)
        deltas = self.data[:-1, :-1, :] - self.data[1:, 1:, :]

        dh, dw, dd = deltas.shape
        deltas = np.reshape(deltas, (dh * dw, dd))
        noise.cov = invertible_cov(deltas) / 2.0

        self.data = spy.mnf(signal, noise).denoise(self.data, snr=self.denoise_snr)
        assert is_masked_cube(self.data)

    def __str__(self) -> str:
        return f"""MinimumNoiseFractionTransform(
            denoise_snr = {self.denoise_snr}
        )"""
