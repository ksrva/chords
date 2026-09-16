"""Chroma tests.

Every tolerance here was measured before it was asserted -- see the comment on
each one for the observed value. Thresholds are set loose enough not to be
brittle and tight enough that a real regression trips them.
"""

from __future__ import annotations

import numpy as np
import pytest

from chordrec.chroma import (
    chroma,
    frame_signal,
    frame_times,
    normalize_frames,
    pitch_class_matrix,
    semitone_resolution,
    stft_magnitude,
)
from chordrec.signals import progression, triad

SR = 22050
N_FFT = 4096
HOP = 2048
PC = "C C# D D# E F F# G G# A A# B".split()

MAJOR_SET = {0: {0, 4, 7}}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(denom > 0, np.sum(a * b, axis=1) / np.maximum(denom, 1e-12), 0.0)


# --------------------------------------------------------------------------
# Agreement with librosa (milestone 2's acceptance criterion)
# --------------------------------------------------------------------------

def test_chroma_agrees_with_librosa():
    """Our chroma tracks librosa's closely but is not identical, by design.

    librosa weights octaves with a Gaussian centred near C5 (ctroct/octwidth);
    ours uses a flat band with hard semitone assignment. So exact equality is
    the wrong test -- the right one is that the two describe the same harmony.

    Measured on a I-vi-IV-V progression: cosine mean 0.974, min 0.971,
    argmax agreement 0.966.
    """
    librosa = pytest.importorskip("librosa")

    y, _ = progression([("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")], 2.0, SR)

    mine = chroma(y, SR, n_fft=N_FFT, hop=HOP, gamma=None, norm="inf", center=True, power=2.0)
    ref = librosa.feature.chroma_stft(
        y=y.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
        tuning=0.0, norm=np.inf, center=True,
    ).T

    n = min(len(mine), len(ref))
    sim = cosine_rows(mine[:n], ref[:n])

    assert sim.min() > 0.95, f"worst frame {sim.min():.4f}"
    assert sim.mean() > 0.96, f"mean {sim.mean():.4f}"
    assert np.mean(mine[:n].argmax(1) == ref[:n].argmax(1)) > 0.90


# --------------------------------------------------------------------------
# Does it actually hear chords?
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "root,quality,expected",
    [
        ("C", "maj", {"C", "E", "G"}),
        ("A", "min", {"A", "C", "E"}),
        ("F", "maj", {"F", "A", "C"}),
        ("G", "maj", {"G", "B", "D"}),
        ("D", "min", {"D", "F", "A"}),
    ],
)
def test_triad_top_three_pitch_classes(root, quality, expected):
    """The three strongest pitch classes are the three notes of the triad.

    Note what is *not* asserted: that the root is the strongest. It usually is
    not. A note's third partial is a twelfth above it, so the fifth of a chord
    collects energy from the root as well as from itself and routinely wins --
    C major reads G > E > C here. That is not a bug, it is what octave-folded
    spectra look like, and it is why the detector scores whole templates rather
    than picking a root. If a future change makes the root dominant, that is a
    real behavioural change worth noticing, not an improvement to assume.
    """
    v = chroma(triad(root, quality, 2.0, SR), SR, n_fft=N_FFT, hop=HOP).mean(0)
    top3 = {PC[i] for i in np.argsort(v)[::-1][:3]}
    assert top3 == expected, f"got {top3}, energies {dict(zip(PC, v.round(3)))}"


def test_major_and_minor_are_distinguishable():
    """C major and C minor differ where they should: the third."""
    maj = chroma(triad("C", "maj", 2.0, SR), SR, n_fft=N_FFT, hop=HOP).mean(0)
    mnr = chroma(triad("C", "min", 2.0, SR), SR, n_fft=N_FFT, hop=HOP).mean(0)
    assert maj[4] > maj[3]   # E over Eb
    assert mnr[3] > mnr[4]   # Eb over E


# --------------------------------------------------------------------------
# Framing, normalization, band limits
# --------------------------------------------------------------------------

def test_causal_framing_uses_only_past_samples():
    """center=False must never read ahead -- the engine has no future samples."""
    y = np.zeros(SR)
    y[SR // 2:] = 1.0                      # step halfway through
    frames = frame_signal(y, N_FFT, HOP, center=False)
    for i, f in enumerate(frames):
        if i * HOP + N_FFT <= SR // 2:     # frame ends before the step
            assert np.all(f == 0.0), f"frame {i} saw the future"


def test_frame_count_and_times():
    y = np.zeros(SR * 2)
    frames = frame_signal(y, N_FFT, HOP, center=False)
    assert frames.shape == (1 + (len(y) - N_FFT) // HOP, N_FFT)
    t = frame_times(len(frames), SR, HOP, N_FFT, center=False)
    # Causal frames are stamped at the moment they could first be emitted.
    assert t[0] == pytest.approx(N_FFT / SR)
    assert np.all(np.diff(t) > 0)


def test_silence_stays_zero_not_noise():
    """Silent frames must not be normalized up into a spurious chord."""
    out = chroma(np.zeros(SR), SR, n_fft=N_FFT, hop=HOP)
    assert out.shape[1] == 12
    assert np.all(out == 0.0)


@pytest.mark.parametrize("norm,fn", [
    ("l2", lambda r: np.sqrt((r**2).sum())),
    ("l1", lambda r: np.abs(r).sum()),
    ("inf", lambda r: np.abs(r).max()),
])
def test_normalization(norm, fn):
    X = np.abs(np.random.default_rng(0).normal(size=(10, 12))) + 0.1
    for row in normalize_frames(X, norm=norm):
        assert fn(row) == pytest.approx(1.0)


def test_band_limits_exclude_out_of_band_bins():
    M = pitch_class_matrix(SR, N_FFT, fmin_hz=100.0, fmax_hz=1000.0)
    freqs = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    assigned = M.sum(axis=1) > 0
    assert not assigned[freqs < 100.0].any()
    assert not assigned[freqs > 1000.0].any()
    assert assigned[(freqs >= 100.0) & (freqs <= 1000.0)].all()


def test_every_bin_maps_to_exactly_one_pitch_class():
    M = pitch_class_matrix(SR, N_FFT)
    sums = M.sum(axis=1)
    assert set(np.unique(sums)) <= {0.0, 1.0}


def test_tuning_offset_rotates_the_whole_chroma():
    """`tuning` is in semitones and positive means the source is sharp (S6).

    The grid is shifted by subtracting the offset from every bin's MIDI value,
    so tuning=+1.0 moves all content to the pitch class *below*: the chroma
    vector rotates down by one, not up. Asserting the full rotation rather than
    just the peak pins the convention down -- getting this sign backwards would
    silently transpose every chord the detector reports.
    """
    y = triad("C", "maj", 2.0, SR)
    plain = chroma(y, SR, n_fft=N_FFT, hop=HOP).mean(0)
    shifted = chroma(y, SR, n_fft=N_FFT, hop=HOP, tuning=1.0).mean(0)
    assert np.allclose(shifted, np.roll(plain, -1), atol=1e-9)


# --------------------------------------------------------------------------
# The bass-resolution limit, as a number rather than a worry
# --------------------------------------------------------------------------

def test_bass_resolution_limit_is_where_we_think_it_is():
    """Records the STFT bass limit so a parameter change cannot hide it.

    Bins per semitone at 22050 Hz; above 1.0 means a semitone is narrower than
    one bin and the distinction is simply not in the transform:

              C2 (65Hz)   G2 (98Hz)   C3 (131Hz)
        4096     1.38        0.92        0.69
        8192     0.69        0.46        0.35

    So n_fft=4096 cannot resolve semitones below roughly G2, and fixing that
    costs a 8192-sample window -- 372 ms, against a "few hundred ms" p95
    latency budget. That trade is the substance of the S3 sweep.
    """
    assert semitone_resolution(SR, 4096, 65.4) > 1.0    # C2 unresolvable
    assert semitone_resolution(SR, 4096, 98.0) < 1.0    # G2 just resolvable
    assert semitone_resolution(SR, 8192, 65.4) < 1.0    # C2 resolvable at 8192
    assert semitone_resolution(SR, 8192, 65.4) == pytest.approx(0.69, abs=0.02)


def test_stft_shape():
    y = np.zeros(SR)
    S = stft_magnitude(y, n_fft=N_FFT, hop=HOP)
    assert S.shape[1] == N_FFT // 2 + 1
