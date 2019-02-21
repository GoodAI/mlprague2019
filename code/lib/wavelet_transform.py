import numpy as np
import pycwt as wavelet


def continuous_wavelet_transform(data):
    """ Written using the tutorial at https://pycwt.readthedocs.io/en/latest/tutorial.html"""

    dt = 0.25
    dj = 1 / 12
    dat = (data - data.mean()) / data.std()
    s0 = 2 * dt
    J = 7 / dj
    mother = wavelet.Morlet(6)

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat, dt, dj, s0, J, mother)

    power = (np.abs(wave)) ** 2
    power = np.log2(power)
    return power

