#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from hipp.core.module import Module
from hipp.utils.utility import is_masked_cube


class StandardNormalVariate(Module):
    def __init__(self) -> None:
        super().__init__(name=__name__, path=__file__, normalize_output=False)

    def run(self) -> None:
        pixel_means = self.data.mean(axis=-1)[:, :, np.newaxis]
        pixel_stds = self.data.std(axis=-1)[:, :, np.newaxis]
        self.data = (self.data - pixel_means) / pixel_stds

        assert is_masked_cube(self.data)

    def __str__(self) -> str:
        return f"""StandardNormalVariate()"""
