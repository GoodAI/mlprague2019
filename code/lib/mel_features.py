# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This file has been modified by GoodAI

"""Defines routines to compute mel spectrogram features from audio waveform."""

import numpy as np
from scipy.io import wavfile
import resampy
import logging

log = logging.getLogger()


def read_wav(filename, target_sample_rate=16000, verbose=False):
    """Read wav file and convert to numpy array

    Returns:
      1-D np.array with PCM data as np.float32
    """
    if verbose:
        log.info("Reading: \'{}\' ".format(filename), end='')
    rate, data = wavfile.read(filename)

    assert (data.dtype == np.int16)
    
    data = np.float32(data) / 32768.0

    # stereo -> mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to the correct sample rate if necessary
    if rate != target_sample_rate:
        data = resampy.resample(data, rate, target_sample_rate)

    if verbose:
        log.debug("Done! Length: {:.2f}s, Sample rate: {}".format(len(data) / rate, rate))
    return data


def frame(data, window_length, hop_length):
    """Convert array into a sequence of successive possibly overlapping frames.

    An n-dimensional array of shape (num_samples, ...) is converted into an
    (n+1)-D array of shape (num_frames, window_length, ...), where each frame
    starts hop_length points after the preceding one.

    This is accomplished using stride_tricks, so the original data is not
    copied.  However, there is no zero-padding, so any incomplete frames at the
    end are not included.

    Args:
        data: np.array of dimension N >= 1.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.

    Returns:
        (N+1)-D np.array with as many rows as there are complete frames that can be
        extracted.
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
    """Calculate a "periodic" Hann window.

    The classic Hann window is defined as a raised cosine that starts and
    ends on zero, and where every value appears twice, except the middle
    point for an odd-length window.  Matlab calls this a "symmetric" window
    and np.hanning() returns it.  However, for Fourier analysis, this
    actually represents just over one cycle of a period N-1 cosine, and
    thus is not compactly expressed on a length-N Fourier basis.  Instead,
    it's better to use a raised cosine that ends just before the final
    zero value - i.e. a complete cycle of a period-N cosine.  Matlab
    calls this a "periodic" window. This routine calculates it.

    Args:
        window_length: The number of points in the returned window.

    Returns:
        A 1D np.array containing the periodic hann window.
    """
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                               np.arange(window_length)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
    """Calculate the short-time Fourier transform magnitude.

    Args:
        signal: 1D np.array of the input time-domain signal.
        fft_length: Size of the FFT to apply.
        hop_length: Advance (in samples) between each frame passed to FFT.
        window_length: Length of each block of samples to pass to FFT.

    Returns:
        2D np.array where each row contains the magnitudes of the fft_length/2+1
        unique values of the FFT for the corresponding frame of input samples.
    """
    frames = frame(signal, window_length, hop_length)
    # Apply frame window to each frame. We use a periodic Hann (cosine of period
    # window_length) instead of the symmetric Hann of np.hanning (period
    # window_length-1).
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


def spectrogram(data,
                window_length_secs=0.1,
                hop_length_secs=0.01,
                sample_rate=16000,
                logarithmic=False,
                ):
    """ Generate a spectrogram

    NOTES:
        * Hi-fidelity (500k params)
        window_length_secs = 0.064  # smaller == less vertical res
        hop_length_secs = 0.001  # smaller no. == finer res.

        * original (54k params)
        window_length_secs = 0.025
        hop_length_secs = 0.01

        Test output:
        plt.imshow(np.flipud(get_mel(data).T), aspect='auto')

    :param data:                Raw PCM signal
    :param sample_rate:         Signal of original data source
    :param window_length_secs:  Length of window used by FFT in seconds
    :param hop_length_secs:     Advance between successive analysis windows.
    : logarithmic:              Should spectrogram be scaled by a log?
    :return:                    Spectrogram image
    """

    # Generate spectrograms
    window_length_samples = int(round(sample_rate * window_length_secs))
    hop_length_samples = int(round(sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    spec = stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples)

    if logarithmic:
        return np.log(spec + 0.01)

    else:
        return spec


# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
    """Convert frequencies to mel scale using HTK formula.

    Args:
        frequencies_hertz: Scalar or np.array of frequencies in hertz.

    Returns:
        Object of same size as frequencies_hertz containing corresponding values
        on the mel scale.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=256,
                              num_spectrogram_bins=1025,
                              sample_rate=16000,
                              lower_edge_hertz=10.0,
                              upper_edge_hertz=5000.0):
    """Return a matrix that can post-multiply spectrogram rows to make mel.

    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
    "mel spectrogram" M of frames x num_mel_bins.  M = S A.

    The classic HTK algorithm exploits the complementarity of adjacent mel bands
    to multiply each FFT bin by only one mel weight, then add it, with positive
    and negative signs, to the two adjacent mel bands to which that bin
    contributes.  Here, by expressing this operation as a matrix multiply, we go
    from num_fft multiplies per frame (plus around 2*num_fft adds) to around
    num_fft^2 multiplies and adds.  However, because these are all presumably
    accomplished in a single call to np.dot(), it's not clear which approach is
    faster in Python.  The matrix multiplication has the attraction of being more
    general and flexible, and much easier to read.

    Args:
        : num_mel_bins:         How many bands in the resulting mel spectrum.  This is
        the number of columns in the output matrix.
        : num_spectrogram_bins: How many bins there are in the source spectrogram
        data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
        only contains the nonredundant FFT bins.
        : sample_rate:          Samples per second of the audio at the input to the
        spectrogram. We need this to figure out the actual frequencies for
        each spectrogram bin, which dictates how they are mapped into mel.
        : lower_edge_hertz:     Lower bound on the frequencies to be included in
        the mel spectrum.  This corresponds to the lower edge of the lowest triangular
        band.
        : upper_edge_hertz:     The desired top edge of the highest frequency band.

    Returns:
        An np.array with shape (num_spectrogram_bins, num_mel_bins).

    Raises:
        ValueError: if frequency edges are incorrectly ordered.
    """
    nyquist_hertz = sample_rate / 2.
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                         (lower_edge_hertz, upper_edge_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                                 hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                       (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                       (upper_edge_mel - center_mel))
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                              upper_slope))
    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def mel_spectrogram(data,
                    sample_rate,
                    log_offset,
                    window_length_secs,
                    hop_length_secs,
                    logarithmic,
                    num_mel_bins,
                    lower_edge_hertz,
                    upper_edge_hertz
                    ):
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    To visualise the spectrogram, use the following code:
        plt.imshow(np.flipud(spec_to_mel(data).T), aspect='auto')
        plt.show()

    Args:
        : data:             1D np.array of waveform data.
        : sample_rate:      The sampling rate of data.
        : log_offset:       Add this to values when taking log to avoid -Infs.
        :window_length_secs:Duration of each window to analyze.
        : hop_length_secs:  Advance between successive analysis windows.
        : logarithmic:      Should spectrogram be scaled by a log?

        **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.

    Returns:
        2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
        magnitudes for successive frames.
    """
    spec = spectrogram(data, sample_rate=sample_rate, logarithmic=False,
    window_length_secs=window_length_secs, hop_length_secs=hop_length_secs)

    mel_matrix = spectrogram_to_mel_matrix(
        num_spectrogram_bins=spec.shape[1],
        sample_rate=sample_rate,
        num_mel_bins=num_mel_bins,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz
    )
    
    result = np.dot(spec, mel_matrix)
    
    if logarithmic:
        return np.log(result + log_offset)
    
    else:
        return result
