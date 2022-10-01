#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hipp.utils.utility import imshow, normalize

from hipp.core.pipeline import Pipeline
from hipp.io.dataloader import load_envi

from hipp.dimensionality_reduction.spectral_binning import SpectralBinning
from hipp.dimensionality_reduction.principal_components import PrincipalComponents
from hipp.dimensionality_reduction.spatial_binning import SpatialBinning


from hipp.scatter_correction.standard_normal_variate import StandardNormalVariate
from hipp.scatter_correction.multiplicative_scatter_correction import (
    MultiplicativeScatterCorrection,
)

from hipp.denoising.minimum_noise_fraction_transform import (
    MinimumNoiseFractionTransform,
)
from hipp.denoising.savitzky_golay_filter import SavitzkyGolayFilter
from hipp.denoising.wavelet_filter import WaveletFilter

from hipp.outlier_detection.stochastic_outlier_selection import (
    StochasticOutlierSelection,
)
from hipp.outlier_detection.copula_based_outlier_detection import (
    CopulaBasedOutlierDetection,
)
from hipp.outlier_detection.reed_xiaoli_detector import (
    ReedXiaoliDetector,
)

from hipp.dead_pixel_detection.standard_deviation_threshold import (
    StandardDeviationThreshold,
)
from hipp.background_removal.otsu_thresholding import OtsuThresholding
from hipp.spike_detection.standard_deviation_factor import StandardDeviationFactor

if __name__ == "__main__":

    """
    data, envi_header = load_envi(
        "/workspace/Messdurchläufe/EVK/2021-03-31_Lieferung_2/2021-03-31T20:15:25_hohlherzig_4_3/hypercube/hypercube.hdr"
    )
    """

    data, envi_header = load_envi(
        "~/Schreibtisch/2021-03-31T20-16-17_hohlherzig_4_4/hypercube/hypercube.hdr"
    )

    pipeline = Pipeline()

    # pipeline.add(StandardDeviationThreshold())
    # pipeline.add(SpectralBinning())
    # pipeline.add(PrincipalComponents())
    # pipeline.add(OtsuThresholding())
    # pipeline.add(StandardDeviationFactor())
    # pipeline.add(StandardNormalVariate())
    # pipeline.add(MultiplicativeScatterCorrection())
    # pipeline.add(SavitzkyGolayFilter())
    # pipeline.add(StochasticOutlierSelection())
    # pipeline.add(CopulaBasedOutlierDetection())
    # pipeline.add(SpatialBinning())
    # pipeline.add(ReedXiaoliDetector())
    # pipeline.add(OtsuThresholding())
    # pipeline.add(WaveletFilter())
    pipeline.add(MinimumNoiseFractionTransform())

    clean_data = pipeline.run(data)

    assert not (clean_data == data).all()

    clean_data = normalize(clean_data)
    print(clean_data.shape)

    imshow(clean_data[..., 0])
