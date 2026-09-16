"""The baseline detector, end to end, and the first number.

chroma -> template cosine -> frame-wise argmax -> intervals -> WCSR.

No transition model, no smoothing, no learning. This exists to be beaten, and
to make "the HMM helped" a measured delta rather than a belief.

    python -m chordrec.baseline
"""

from __future__ import annotations

import numpy as np

from .chroma import chroma
from .evaluate import frame_intervals, score, states_to_intervals, transitions_per_minute
from .signals import progression
from .templates import emission_scores, predict_states
from .vocab import NO_CHORD_INDEX

SR = 22050
N_FFT = 4096
HOP = 2048


def detect(
    y: np.ndarray,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop: int = HOP,
    no_chord_threshold: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """Audio in, chord intervals out."""
    X = chroma(y, sr, n_fft=n_fft, hop=hop)
    states = predict_states(X, no_chord_threshold=no_chord_threshold)
    return states_to_intervals(states, frame_intervals(len(states), sr, hop, n_fft))


def add_noise(y: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """White noise at a given SNR.

    Kept in the table despite being almost useless as a stressor, because the
    uselessness is the point: white noise is spectrally flat, so it lands on
    all twelve pitch classes equally and per-frame normalization divides it
    straight back out. Measured WCSR is unchanged from clean down to 0 dB. Any
    robustness claim built on white noise would be worthless, and the table
    should say so out loud rather than quietly omit it.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=len(y))
    scale = np.sqrt(np.mean(y**2) / (np.mean(noise**2) * 10 ** (snr_db / 10.0)))
    return y + scale * noise


def add_percussion(y: np.ndarray, sr: int = SR, bpm: float = 120.0,
                   level: float = 0.5, seed: int = 0) -> np.ndarray:
    """Broadband clicks on the beat -- a crude drum track.

    Unlike white noise this is concentrated in time rather than spread through
    it, so it corrupts whole frames instead of adding a constant to every
    pitch class. That is how real percussion breaks a chroma front end, and it
    hits hardest exactly at chord boundaries, where the beat lands.
    """
    rng = np.random.default_rng(seed)
    out = y.copy()
    click_len = int(0.02 * sr)
    envelope = np.exp(-np.linspace(0, 8, click_len))
    for start in range(0, len(y) - click_len, int(sr * 60.0 / bpm)):
        out[start:start + click_len] += level * envelope * rng.normal(size=click_len)
    return out


def add_interferer(y: np.ndarray, semitones: float, sr: int = SR,
                   level: float = 0.6) -> np.ndarray:
    """A sustained out-of-chord note -- a bass line or melody on a wrong note.

    Harmonic interference is the stressor that actually flips rankings, because
    it puts energy on one specific pitch class rather than all of them.
    """
    from .signals import midi_to_hz, tone
    return y + level * tone(midi_to_hz(48 + semitones), len(y) / sr, sr)


def main() -> None:
    chords = [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj"),
              ("D", "min"), ("G", "maj"), ("C", "maj"), ("E", "min")]
    y, ref = progression(chords, seconds_each=2.0, sr=SR)
    ref_intervals = np.array([[s, e] for s, e, _ in ref])
    ref_labels = [f"{r}:{q}" for _, _, lab in ref for r, q in [lab.split(":")]]
    ref_cpm = transitions_per_minute(ref_intervals)

    print("baseline: chroma + templates + frame-wise argmax")
    print("synthetic audio only -- this measures plumbing, not accuracy on music")
    print()
    print(f"  {'condition':<18} {'WCSR':>6} {'root':>6} {'thirds':>7} "
          f"{'chords':>7} {'chg/min':>8} {'margin':>7}")
    print("  " + "-" * 64)

    conditions = [
        ("clean", lambda a: a),
        ("white noise 10dB", lambda a: add_noise(a, 10.0)),
        ("white noise 0dB", lambda a: add_noise(a, 0.0)),
        ("percussion", lambda a: add_percussion(a)),
        ("percussion loud", lambda a: add_percussion(a, level=2.0)),
        # Levels are relative to the music's RMS, calibrated by sweep: below
        # ~0.35x nothing happens, above ~1.5x the interferer simply *is* the
        # signal and a score of 0.000 measures nothing interesting. The
        # collapse happens in between, so that is where the rows are.
        ("interferer 0.35x", lambda a: add_interferer(a, 6.0, level=0.10)),
        ("interferer 0.71x", lambda a: add_interferer(a, 6.0, level=0.20)),
        ("interferer 1.06x", lambda a: add_interferer(a, 6.0, level=0.30)),
    ]
    for name, degrade in conditions:
        audio = degrade(y)
        X = chroma(audio, SR, n_fft=N_FFT, hop=HOP)
        ranked = np.sort(emission_scores(X)[:, :NO_CHORD_INDEX], axis=1)
        margin = float(np.mean(ranked[:, -1] - ranked[:, -2]))

        est_intervals, est_labels = detect(audio)
        r = score(ref_intervals, ref_labels, est_intervals, est_labels)
        cpm = transitions_per_minute(est_intervals)
        print(f"  {name:<18} {r['wcsr']:6.3f} {r['root']:6.3f} {r['thirds']:7.3f} "
              f"{r['n_est']:7d} {cpm:8.0f} {margin:7.3f}")

    print(f"  {'reference':<18} {1.0:6.3f} {1.0:6.3f} {1.0:7.3f} "
          f"{len(ref_labels):7d} {ref_cpm:8.0f} {'-':>7}")
    print()
    print("  chg/min is the flicker diagnostic: a memoryless detector re-decides every")
    print("  frame, and the gap from the reference is what the HMM has to close.")
    print("  margin is the mean gap between the best chord and the runner-up. It is")
    print("  thin even when clean, which is why WCSR can hold while confidence does not.")
    print()
    print("  white noise and percussion barely register: both spread energy across all")
    print("  twelve pitch classes, and per-frame normalization divides that back out.")
    print("  Harmonic interference is what actually breaks it, because it lands on one")
    print("  pitch class. Robustness claims built on added noise would be worthless.")


if __name__ == "__main__":
    main()
