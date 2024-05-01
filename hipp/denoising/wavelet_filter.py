import numpy.ma as ma

from hipp.core.module import Module
from skimage.restoration import denoise_wavelet


class WaveletFilter(Module):
    def __init__(self, wavelet="db12", multichannel=True) -> None:
        self.wavelet = wavelet
        self.multichannel = multichannel
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        self.data = ma.masked_array(
            denoise_wavelet(
                self.data,
                wavelet=self.wavelet,
                multichannel=self.multichannel,
                mode="soft",
                method="BayesShrink",
            ),
            mask=self.data.mask,
            fill_value=0,
        )

    def __str__(self) -> str:
        return f"""WaveletFilter(
            wavelet = {self.wavelet},
            multichannel = {self.multichannel}
        )"""
