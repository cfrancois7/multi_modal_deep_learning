from audiomentations.core.transforms_interface import BaseWaveformTransform
import numpy.typing as npt
import numpy as np
from numpy import random


class AddRandomStationaryNoise(BaseWaveformTransform):
    """Add some stationary sinusoid(s) to the samples"""

    supports_multichannel = True

    def __init__(
        self,
        min_frequency: float = 0.001,
        max_frequency: float = 16000.0,
        min_amplitude: float = 0.001,
        max_amplitude: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__(p)
        assert min_frequency > 0.0
        assert max_frequency <= 16000.0
        assert min_frequency < max_frequency
        assert min_amplitude < max_amplitude
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude

    def randomize_parameters(
        self, samples: npt.NDArray[np.float_], sample_rate: int
    ) -> None:
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["amplitude"] = random.uniform(
                self.min_amplitude, self.max_amplitude
            )
            self.parameters["frequency"] = random.uniform(
                self.min_frequency, self.max_frequency
            )
            self.parameters["phase"] = 2 * np.pi * random.uniform(0, 1)

    def apply(
        self, samples: npt.NDArray[np.float_], sample_rate: int
    ) -> npt.NDArray[np.float_]:
        p = self.parameters["phase"]
        f = self.parameters["frequency"]
        a = self.max_amplitude
        if len(samples.shape) > 1:
            t = np.arange(samples.shape[1])
        else:
            t = np.arange(samples.shape[0])
        noise = a * np.sin(t * f + p)
        samples = samples + noise.astype(float)
        return samples
