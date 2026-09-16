"""Evaluation: turn frame predictions into intervals and score them (M5).

Weighted chord symbol recall is the headline metric -- the fraction of a song's
*duration* (not of its frames, and not of its chords) that is labelled
correctly. Duration weighting is what stops a system being rewarded for
nailing a hundred fast passing chords while missing the four-bar tonic.

mir_eval does the comparison. What this module owns is everything on the way
in, which is where the errors actually live.

The timestamp question
----------------------
A frame is computed from a window of audio, so which instant does it describe?
Two defensible answers, needed for two different purposes, and conflating them
quietly corrupts either the accuracy numbers or the latency numbers:

  * For **evaluation**, a frame describes the middle of its window. Frame i
    covers samples [i*hop, i*hop + n_fft), so it is centred at
    i*hop + n_fft/2. Using the window's *end* instead would shift every
    prediction late by half a window -- 93 ms at n_fft=4096 -- and quietly cost
    a few points of WCSR that would then be blamed on the algorithm.

  * For **latency** (M6/S2), what matters is when the engine could first emit
    the frame, which is the window's *end*: i*hop + n_fft. That is
    `chroma.frame_times`, and it is not what evaluation uses.

This module implements the first. The two must not be merged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .vocab import NO_CHORD_LABEL, reduce_label, state_label


def frame_intervals(n_frames: int, sr: int, hop: int, n_fft: int) -> np.ndarray:
    """(n_frames, 2) start/end times, each frame centred on its window.

    Boundaries are computed once and sliced, so interval i's end is bit-identical
    to interval i+1's start. Computing each endpoint independently looks
    equivalent and is not: rounding makes some starts land a few times 1e-15
    *below* the previous end, and mir_eval compares exactly and rejects the
    whole annotation with "Chord Intervals must not overlap". Asserting the
    tiling with np.allclose hides this; the test asserts exact equality.
    """
    bounds = (np.arange(n_frames + 1, dtype=np.float64) * hop
              + n_fft / 2.0 - hop / 2.0) / sr
    bounds = np.maximum(bounds, 0.0)
    return np.stack([bounds[:-1], bounds[1:]], axis=1)


def states_to_intervals(
    states: np.ndarray,
    intervals: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Collapse per-frame states into contiguous labelled intervals."""
    states = np.asarray(states)
    if len(states) == 0:
        return np.zeros((0, 2)), []

    change = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(states)]])

    out = np.stack([intervals[starts, 0], intervals[ends - 1, 1]], axis=1)
    return out, [state_label(int(states[s])) for s in starts]


def read_lab(path: str | Path, reduce: bool = True) -> tuple[np.ndarray, list[str]]:
    """Read a `.lab` annotation: `start end label` per line, whitespace split.

    With `reduce=True` every label is pushed through the major/minor reduction,
    so what comes back is always inside the 25-state vocabulary.
    """
    starts, ends, labels = [], [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"malformed line in {path}: {line!r}")
        starts.append(float(parts[0]))
        ends.append(float(parts[1]))
        label = " ".join(parts[2:])
        labels.append(state_label(reduce_label(label)) if reduce else label)
    return np.stack([starts, ends], axis=1) if starts else np.zeros((0, 2)), labels


def write_lab(path: str | Path, intervals: np.ndarray, labels: list[str]) -> None:
    lines = [f"{s:.6f}\t{e:.6f}\t{lab}" for (s, e), lab in zip(intervals, labels)]
    Path(path).write_text("\n".join(lines) + "\n")


def score(
    ref_intervals: np.ndarray,
    ref_labels: list[str],
    est_intervals: np.ndarray,
    est_labels: list[str],
) -> dict[str, float]:
    """WCSR and friends, via mir_eval.

    `wcsr` is an alias for mir_eval's `majmin`: the duration-weighted fraction
    correct over the major/minor vocabulary, which is the metric in section 9.
    `root` and `seg` come along because they diagnose *how* a system fails --
    a high root score with a low majmin means thirds are being confused, which
    points at the front end rather than at the transition model.
    """
    import mir_eval

    if len(est_labels) == 0:
        return {"wcsr": 0.0, "root": 0.0, "seg": 0.0, "n_ref": len(ref_labels), "n_est": 0}

    results = mir_eval.chord.evaluate(
        np.asarray(ref_intervals, dtype=float), list(ref_labels),
        np.asarray(est_intervals, dtype=float), list(est_labels),
    )
    return {
        "wcsr": float(results["majmin"]),
        "root": float(results["root"]),
        "thirds": float(results["thirds"]),
        "seg": float(results["seg"]),
        "n_ref": len(ref_labels),
        "n_est": len(est_labels),
    }


def transitions_per_minute(intervals: np.ndarray) -> float:
    """Chord changes per minute -- the flicker diagnostic.

    A frame-wise detector with no memory produces wild over-segmentation, and
    this number makes that visible in a way WCSR alone does not. Real songs sit
    somewhere around 10-40; a baseline emitting hundreds is telling you exactly
    what the transition model is for.
    """
    if len(intervals) == 0:
        return 0.0
    span = float(intervals[-1, 1] - intervals[0, 0])
    return len(intervals) / (span / 60.0) if span > 0 else 0.0
