#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hipp.core.pipeline import Pipeline
from hipp.io.dataloader import load_envi
from hipp.utils.utility import imshow, normalize

# Background removal method
from hipp.background_removal.otsu_thresholding import OtsuThresholding

# Dead pixel removal method
from hipp.dead_pixel_detection.standard_deviation_threshold import (
    StandardDeviationThreshold,
)

# Spike removal method
from hipp.spike_detection.standard_deviation_factor import StandardDeviationFactor

# Outlier detection methods
from hipp.outlier_detection.stochastic_outlier_selection import (
    StochasticOutlierSelection,
)
from hipp.outlier_detection.copula_based_outlier_detection import (
    CopulaBasedOutlierDetection,
)
from hipp.outlier_detection.reed_xiaoli_detector import (
    ReedXiaoliDetector,
)

# Noise reduction methods
from hipp.denoising.minimum_noise_fraction_transform import (
    MinimumNoiseFractionTransform,
)
from hipp.denoising.savitzky_golay_filter import SavitzkyGolayFilter
from hipp.denoising.wavelet_filter import WaveletFilter

# Scatter correction methods
from hipp.scatter_correction.standard_normal_variate import StandardNormalVariate
from hipp.scatter_correction.multiplicative_scatter_correction import (
    MultiplicativeScatterCorrection,
)

# Dimensionality reduction methods
from hipp.dimensionality_reduction.spectral_binning import SpectralBinning
from hipp.dimensionality_reduction.principal_components import PrincipalComponents
from hipp.dimensionality_reduction.spatial_binning import SpatialBinning


if __name__ == "__main__":
    # Load hypercube using ENVI header file
    data, envi_header = load_envi("hypercube.hdr")
    
    # Instantiate empty processing pipeline
    pipeline = Pipeline()

    # Background removal
    pipeline.add(OtsuThresholding())
    
    # Dead pixel removal
    pipeline.add(StandardDeviationThreshold())

    # Spike removal
    pipeline.add(StandardDeviationFactor())

    # Outlier detection
    pipeline.add(ReedXiaoliDetector())

    # Noise reduction
    pipeline.add(MinimumNoiseFractionTransform())
    
    # Scatter correction
    pipeline.add(StandardNormalVariate())
    
    # Dimensionality reduction (use with care)
    # pipeline.add(PrincipalComponents())
    
    # Execute the preprocessing pipeline
    preprocessed_data = pipeline.run(data)

    # Image processing: May start with data normalization
    preprocessed_data = normalize(preprocessed_data_data)
    
    # Visualize first preprocessed spectral band
    imshow(preprocessed_data[..., 0])
