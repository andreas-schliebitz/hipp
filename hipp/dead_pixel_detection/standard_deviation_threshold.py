#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from hipp.core.module import Module
from hipp.utils.utility import interpolate, is_masked_cube


class StandardDeviationThreshold(Module):
    def __init__(self, dead_std_threshold: float = 0.001) -> None:
        assert 0 < dead_std_threshold <= 1

        self.dead_std_threshold = dead_std_threshold
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        for z in range(self.data.shape[-1]):
            self.data[:, :, z] = np.apply_along_axis(
                interpolate,
                1,
                self.data[:, :, z],
                self.data[:, :, z].std(axis=0) < self.dead_std_threshold,
            )
        assert is_masked_cube(self.data)

    def __str__(self) -> str:
        return f"""StandardDeviationThreshold(
            dead_std_threshold = {self.dead_std_threshold}
        )"""
