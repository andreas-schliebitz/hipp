#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma

from numba import njit
from collections import defaultdict
from typing import List, Tuple

from hipp.core.module import Module


class SpectralBinning(Module):
    def __init__(
        self,
        band_ranges: List[Tuple] = None,
        min_band_binning_distance: float = None,
        mean_distance_deviation_perc: float = 0.25,
    ) -> None:
        self.band_ranges = band_ranges
        self.min_band_binning_distance = min_band_binning_distance
        self.mean_distance_deviation_perc = mean_distance_deviation_perc

        super().__init__(name=__name__, path=__file__, normalize_output=False)

    def run(self) -> None:
        if not self.min_band_binning_distance:
            self.min_band_binning_distance = np.abs(
                np.diff(self.data.mean(axis=(0, 1)))
            ).mean()

        assert (
            self.min_band_binning_distance > 0
            and 0 <= self.mean_distance_deviation_perc <= 1
        )

        self.min_band_binning_distance += (
            self.min_band_binning_distance * self.mean_distance_deviation_perc
        )

        if self.band_ranges:
            data = _bin_by_band_ranges(self.data, self.band_ranges)
        else:
            data = self._bin_by_threshold()

        self.data = ma.masked_array(
            data,
            mask=np.repeat(self.data.mask[:, :, 0], data.shape[-1], axis=-1),
            fill_value=0,
        )

    def _bin_by_threshold(self) -> np.ndarray:
        band_means = np.mean(self.data, axis=(0, 1))

        if not self.min_band_binning_distance:
            assert (
                self.min_band_binning_distance > 0
                and 0 <= self.mean_distance_deviation_perc <= 1
            )
            self.min_band_binning_distance = np.abs(np.diff(band_means)).mean()
            self.min_band_binning_distance += (
                self.min_band_binning_distance * self.mean_distance_deviation_perc
            )

        bin_means = np.zeros_like(band_means)
        bins = defaultdict(list)
        bin_index = 0
        for n, cur_mean in enumerate(band_means):
            if n == 0 or (
                np.abs(cur_mean - bin_means[bin_index]) < self.min_band_binning_distance
            ):
                bin_means[bin_index] += cur_mean
                bin_means[bin_index] /= len(bins[bin_index]) + 1
            else:
                bin_index += 1
                bin_means[bin_index] = cur_mean
            bins[bin_index].append(self.data[..., n])

        binned_cube = []
        for sub_cube in bins.values():
            sub_cube = np.asarray(sub_cube).transpose((1, -1, 0))
            binned_band = np.mean(sub_cube, axis=2)
            binned_cube.append(binned_band)

        return np.asarray(binned_cube).transpose((1, -1, 0))

    def __str__(self) -> str:
        return f"""SpectralBinning(
            band_ranges = {self.band_ranges}
            min_band_binning_distance = {self.min_band_binning_distance}
            mean_distance_deviation_perc = {self.mean_distance_deviation_perc}
        )"""


@njit(cache=True)
def _bin_by_band_ranges(data: np.ndarray, band_ranges: List[Tuple]) -> np.ndarray:
    binned_cube = []
    for band_range_index in range(len(band_ranges)):
        start = band_range[band_range_index]
        end = band_range[1]
        sub_cube = data[:, :, start:end]
        binned_band = np.mean(sub_cube, axis=2)
        binned_cube.append(binned_band)
    return np.asarray(binned_cube)
