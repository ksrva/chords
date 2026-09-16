"""Chroma: audio in, twelve pitch-class energies per frame out.

This is the front end of the detector and it is written from scratch, which
means every step below is a choice that has to be defended rather than a
library default that happened to be there. The steps are:

    frame -> window -> rFFT -> magnitude -> pitch-class binning
          -> log compression -> per-frame normalization

Only the FFT itself is borrowed (M1 permits this); everything around it is
here. The module is deliberately plain NumPy with no librosa import, because
this code is the reference implementation that the Rust core has to reproduce
within tolerance, and anything clever here becomes something painful there.

Two conventions differ from the usual Python audio stack on purpose:

`center=False` by default. librosa pads the signal by half a window so that
frame *i* is centred on sample *i*hop*. A real-time engine cannot do that: it
has not got the future samples. Framing here is causal by default so that the
offline reference and the online engine see identical windows, and the
comparison tests turn centring back on only to line up with librosa.

Frames are rows. Output is (n_frames, 12), not (12, n_frames). The engine
processes one frame at a time and a row is what one step produces.
"""

from __future__ import annotations

import numpy as np

A4_HZ = 440.0
A4_MIDI = 69.0

# Default analysis band. The bottom is around C2; below that an STFT at any
# window we can afford in real time cannot resolve a semitone (see the note in
# `pitch_class_matrix`). The top is around C7, above which there is little but
# noise and cymbal wash to bin.
DEFAULT_FMIN_HZ = 65.0
DEFAULT_FMAX_HZ = 2100.0

# Compression constant. log(1 + gamma*S) with gamma=100 is the standard choice
# in FMP; it is a free parameter and belongs in the sweep.
DEFAULT_GAMMA = 100.0


def hann(n: int) -> np.ndarray:
    """Periodic Hann window.

    Periodic, not symmetric: the symmetric variant (`np.hanning`) repeats its
    endpoint and gives slightly worse overlap-add behaviour. This matches what
    every STFT implementation actually uses.
    """
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def frame_signal(y: np.ndarray, n_fft: int, hop: int, center: bool = False) -> np.ndarray:
    """Slice a signal into overlapping frames, one per row.

    With `center=True` the signal is reflect-padded by n_fft//2 so frame i is
    centred on sample i*hop, matching librosa. With `center=False` frame i
    *starts* at sample i*hop and uses only past samples, which is the only
    thing a live engine can do.
    """
    y = np.asarray(y, dtype=np.float64)
    if center:
        y = np.pad(y, n_fft // 2, mode="reflect")
    if len(y) < n_fft:
        return np.zeros((0, n_fft), dtype=np.float64)
    n_frames = 1 + (len(y) - n_fft) // hop
    # as_strided would avoid the copy, but this is the reference implementation
    # and a clear loop is worth more here than the memory.
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return y[idx]


def stft_magnitude(
    y: np.ndarray,
    n_fft: int = 4096,
    hop: int = 2048,
    center: bool = False,
    power: float = 2.0,
) -> np.ndarray:
    """Magnitude (or power) spectrogram, shape (n_frames, n_fft//2 + 1)."""
    frames = frame_signal(y, n_fft, hop, center=center)
    if frames.shape[0] == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float64)
    spectrum = np.fft.rfft(frames * hann(n_fft)[None, :], axis=1)
    return np.abs(spectrum) ** power


def hz_to_midi(f: np.ndarray | float) -> np.ndarray | float:
    """Frequency in Hz to (fractional) MIDI note number."""
    return A4_MIDI + 12.0 * np.log2(np.asarray(f, dtype=np.float64) / A4_HZ)


def bin_frequencies(sr: int, n_fft: int) -> np.ndarray:
    """Centre frequency of each rFFT bin."""
    return np.arange(n_fft // 2 + 1, dtype=np.float64) * sr / n_fft


def semitone_resolution(sr: int, n_fft: int, f_hz: float) -> float:
    """Semitones per FFT bin at `f_hz`. Above 1.0 means unresolvable.

    This is the bass-resolution risk in the PRD, made measurable. A semitone at
    f spans f*(2**(1/12) - 1) Hz, and a bin is sr/n_fft wide, so the ratio says
    how many semitones one bin covers. At sr=22050, n_fft=4096 the bin is
    5.4 Hz while a semitone at C2 (65 Hz) is only 4.0 Hz: the root of a bass
    note lands inside a single bin and its neighbours, and no amount of
    downstream cleverness recovers it.
    """
    bin_width = sr / n_fft
    semitone_width = f_hz * (2.0 ** (1.0 / 12.0) - 1.0)
    return bin_width / semitone_width


def pitch_class_matrix(
    sr: int,
    n_fft: int,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    tuning: float = 0.0,
) -> np.ndarray:
    """Binary (n_bins, 12) map from FFT bin to pitch class.

    Hard assignment: each bin in the band is rounded to the nearest semitone
    and credited entirely to that pitch class. This is the simplest defensible
    mapping and therefore the right starting point -- soft/Gaussian weighting
    across neighbouring semitones is a variant to measure against it, not to
    assume is better.

    `tuning` shifts the whole grid by a fraction of a semitone, for instruments
    that are not at A440 (S6). Positive means the source is sharp of concert.
    """
    freqs = bin_frequencies(sr, n_fft)
    n_bins = len(freqs)
    mapping = np.zeros((n_bins, 12), dtype=np.float64)

    with np.errstate(divide="ignore"):
        midi = hz_to_midi(np.maximum(freqs, 1e-12)) - tuning

    in_band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    nearest = np.rint(midi).astype(int)
    for k in np.flatnonzero(in_band):
        mapping[k, nearest[k] % 12] = 1.0
    return mapping


def normalize_frames(X: np.ndarray, norm: str = "l2", eps: float = 1e-8) -> np.ndarray:
    """Normalize each row, leaving near-silent rows as zeros.

    Silent frames are left at zero rather than forced to a unit vector: a zero
    row is an honest "no energy here", and the no-chord state downstream can
    act on it instead of scoring noise against 24 templates.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return X
    if norm == "l2":
        mag = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    elif norm == "l1":
        mag = np.sum(np.abs(X), axis=1, keepdims=True)
    elif norm == "inf":
        mag = np.max(np.abs(X), axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown norm: {norm!r}")
    out = np.zeros_like(X)
    live = (mag > eps).ravel()
    out[live] = X[live] / mag[live]
    return out


def chroma(
    y: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop: int = 2048,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    tuning: float = 0.0,
    gamma: float | None = DEFAULT_GAMMA,
    norm: str = "l2",
    center: bool = False,
    power: float = 2.0,
) -> np.ndarray:
    """Full chroma front end. Returns (n_frames, 12), pitch class 0 = C.

    `gamma=None` disables log compression, which is needed to compare against
    librosa's uncompressed chroma_stft.
    """
    S = stft_magnitude(y, n_fft=n_fft, hop=hop, center=center, power=power)
    if S.shape[0] == 0:
        return np.zeros((0, 12), dtype=np.float64)

    C = S @ pitch_class_matrix(sr, n_fft, fmin_hz, fmax_hz, tuning)
    if gamma is not None:
        C = np.log1p(gamma * C)
    return normalize_frames(C, norm=norm)


def frame_times(n_frames: int, sr: int, hop: int, n_fft: int, center: bool = False) -> np.ndarray:
    """Timestamp of each frame, in seconds.

    Causal frames are timestamped at their *end*: that is the moment the engine
    could first have emitted them, which is what the latency budget in M6 has
    to be measured against. Centred frames are timestamped at their centre, as
    librosa does.
    """
    i = np.arange(n_frames, dtype=np.float64)
    if center:
        return i * hop / sr
    return (i * hop + n_fft) / sr
