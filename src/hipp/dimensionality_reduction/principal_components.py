#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma

from spectral import principal_components
from hipp.core.module import Module
from hipp.utils.utility import is_masked_cube


class PrincipalComponents(Module):
    def __init__(self, retain_variance_perc: float = 0.998) -> None:
        assert 0 < retain_variance_perc < 1

        self.retain_variance_perc = retain_variance_perc
        super().__init__(name=__name__, path=__file__, normalize_output=False)

    def run(self) -> None:
        pc = principal_components(self.data)
        pc_reduce = pc.reduce(fraction=self.retain_variance_perc)
        reduced_data = pc_reduce.transform(self.data)

        if len(reduced_data.shape) == 2:
            reduced_data = reduced_data[:, :, np.newaxis]

        self.data = ma.masked_array(
            reduced_data,
            mask=np.repeat(self.data.mask[..., 0], reduced_data.shape[-1], axis=-1),
            fill_value=0,
        )

        assert is_masked_cube(self.data)
        return pc_reduce.eigenvalues, pc.cov

    def __str__(self) -> str:
        return f"""PrincipalComponents(
            retain_variance_perc = {self.retain_variance_perc}
        )"""
