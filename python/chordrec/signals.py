"""Synthetic test signals.

Ground truth without a single audio file: if we build the waveform, we know
exactly which pitch classes are in it. These carry the chroma tests and will
later carry the Rust-vs-Python fixtures (M8), where the point is agreement
between two implementations rather than realism.
"""

from __future__ import annotations

import numpy as np

A4_HZ = 440.0

# Semitones above C for each triad member.
MAJOR = (0, 4, 7)
MINOR = (0, 3, 7)

NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def midi_to_hz(m: float) -> float:
    return A4_HZ * 2.0 ** ((m - 69.0) / 12.0)


def tone(
    freq: float,
    seconds: float,
    sr: int,
    n_partials: int = 6,
    decay: float = 0.7,
) -> np.ndarray:
    """A note with a harmonic series, amplitudes falling as decay**k.

    A bare sine is unrealistically kind to a chroma front end -- real
    instruments put energy on the fifth and the third via their partials, which
    is precisely what makes major/minor confusable. Partials are included so
    the tests exercise that.
    """
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    out = np.zeros_like(t)
    for k in range(1, n_partials + 1):
        f = freq * k
        if f >= sr / 2:
            break
        out += (decay ** (k - 1)) * np.sin(2.0 * np.pi * f * t)
    return out


def triad(
    root: str,
    quality: str = "maj",
    seconds: float = 2.0,
    sr: int = 22050,
    octave: int = 4,
    **kw,
) -> np.ndarray:
    """A sustained triad. Root pitch class, plus third and fifth above it."""
    intervals = MAJOR if quality == "maj" else MINOR
    root_midi = 12 * (octave + 1) + NOTE_TO_PC[root]
    sig = sum(tone(midi_to_hz(root_midi + i), seconds, sr, **kw) for i in intervals)
    return sig / np.max(np.abs(sig))


def progression(
    chords: list[tuple[str, str]],
    seconds_each: float = 2.0,
    sr: int = 22050,
    **kw,
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """A sequence of triads plus the label intervals they generate.

    Returns (audio, [(start, end, "C:maj"), ...]) -- the same shape as a .lab
    annotation, so evaluation code can be tested before any real data exists.
    """
    parts, labels, t = [], [], 0.0
    for root, quality in chords:
        parts.append(triad(root, quality, seconds_each, sr, **kw))
        labels.append((t, t + seconds_each, f"{root}:{quality}"))
        t += seconds_each
    return np.concatenate(parts), labels
