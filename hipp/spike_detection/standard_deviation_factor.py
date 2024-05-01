import numpy as np
import numpy.ma as ma

from hipp.core.module import Module
from hipp.utils.utility import interpolate, is_masked_cube


class StandardDeviationFactor(Module):
    def __init__(self, spike_std_factor: float = 2) -> None:
        assert 1 < spike_std_factor

        self.spike_std_factor = spike_std_factor
        super().__init__(name=__name__, path=__file__)

    def run(self) -> None:
        spikes = np.abs(self.data - self.data.mean(axis=-1)[:, :, np.newaxis]) > (
            self.spike_std_factor * self.data.std(axis=-1)[:, :, np.newaxis]
        )

        spikes_yx_pos = np.argwhere(spikes)[:, :2]
        _, idx = np.unique(spikes_yx_pos, return_index=True, axis=0)
        spikes_yx_pos = spikes_yx_pos[np.sort(idx)]

        if len(spikes_yx_pos) == 0:
            return

        data = ma.masked_array.copy(self.data)
        for y, x in spikes_yx_pos:
            self.data[y, x, :] = interpolate(data[y, x, :], spikes[y, x, :])

        assert is_masked_cube(self.data)

    def __str__(self) -> str:
        return f"""StandardDeviationFactor(
            spike_std_factor = {self.spike_std_factor}
        )"""
