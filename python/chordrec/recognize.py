"""Recognize chords in a real recording, with memory across frames.

`baseline.detect` decides every frame alone and so has no way to express the
fact that chords last. This module adds the smallest honest amount of memory --
a moving average over the emission scores -- and is the first code in the
project meant to be pointed at an actual piano rather than a synthesised triad.

    python -m chordrec.recognize path/to/piano.wav
    python -m chordrec.recognize --demo

Why this is not the HMM
-----------------------
It is a placeholder that the HMM (M4) has to beat, in the same way templates
are the placeholder that learned emissions have to beat. Averaging scores over
a window says "chords last longer than one frame" and nothing else. An HMM
says that *and* which chords follow which, learned from a corpus, and it
decodes a globally optimal path instead of making each frame's decision
locally. The two are not the same and this must not be allowed to quietly
become the shipped smoother because it was there first.

What it is good for is being measurable now, on real audio, before any corpus
work has happened -- and for establishing the *shape* of the fix, since the
window length here plays the same role the self-transition probability will.

Why the window is causal by default
-----------------------------------
A centred window averages in frames the engine has not heard yet. That is fine
offline and impossible live, and it is exactly the trap `chroma.center=False`
exists to avoid; adopting a centred smoother here would quietly reintroduce the
lookahead that the front end was careful to refuse. So `causal=True` averages
frame i with the `window-1` frames *before* it, and costs latency rather than
correctness: the engine must wait for the window to fill before its answer
stabilises, which is a real number that belongs in the latency budget.

`causal=False` exists to quantify what that decision costs -- run both, take
the difference, and the price of being real-time is on the table instead of
being assumed to be small.
"""

from __future__ import annotations

import numpy as np

from .chroma import chroma
from .evaluate import frame_intervals, states_to_intervals, transitions_per_minute
from .templates import emission_scores
from .vocab import NO_CHORD_INDEX, N_STATES

SR = 22050
N_FFT = 4096
HOP = 2048

# Frames, not seconds, because the smoother operates on frames and hiding the
# conversion would make the latency cost harder to see. At hop=2048, sr=22050
# one frame is 93 ms, so 5 frames is a 464 ms trailing window.
DEFAULT_SMOOTHING_FRAMES = 5

# Segments shorter than this are almost certainly artefacts of the smoother's
# edges rather than chords anyone played. Not a musical claim -- at 120 bpm an
# eighth note is 250 ms -- so it is deliberately below anything musical.
DEFAULT_MIN_DURATION_S = 0.15


def smooth_scores(
    scores: np.ndarray,
    window: int = DEFAULT_SMOOTHING_FRAMES,
    causal: bool = True,
) -> np.ndarray:
    """Moving average over frames of the (n_frames, 25) score matrix.

    Averaging the *scores* rather than the decided labels is the important
    detail. A median or mode filter over state indices is the obvious
    alternative and is wrong here: the 25 states are categorical, and their
    integer ordering (C:maj=0 .. B:min=23, N=24) is an indexing convention, not
    a scale. The median of C:maj and B:min is not a chord halfway between them,
    it is whatever happens to sit at index 11. Averaging scores keeps every
    state's evidence intact and lets a chord that was a close second in two
    neighbouring frames win over one that spiked in a single frame.

    Silent frames carry -inf in every column but N (see `emission_scores`), so
    they are held out of the average entirely rather than poisoning it: one
    silent frame inside the window would otherwise drag every real chord's
    running mean to -inf and hand the segment to N.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != N_STATES:
        raise ValueError(f"expected (n_frames, {N_STATES}), got {S.shape}")
    if window <= 1 or S.shape[0] == 0:
        return S.copy()

    finite = np.isfinite(S)
    values = np.where(finite, S, 0.0)

    # Cumulative sums give a moving average in one pass and, more to the point,
    # the same arithmetic in Rust without a sliding-window buffer.
    pad = np.zeros((1, S.shape[1]))
    csum = np.concatenate([pad, np.cumsum(values, axis=0)], axis=0)
    ccount = np.concatenate([pad, np.cumsum(finite, axis=0)], axis=0)

    n = S.shape[0]
    i = np.arange(n)
    if causal:
        lo = np.maximum(i - window + 1, 0)
        hi = i + 1
    else:
        half = window // 2
        lo = np.maximum(i - half, 0)
        hi = np.minimum(i + half + 1, n)

    total = csum[hi] - csum[lo]
    count = ccount[hi] - ccount[lo]

    out = np.divide(total, count, out=np.full_like(total, -np.inf), where=count > 0)
    # A state that was -inf across the whole window stays -inf: never observed
    # is not the same as observed weakly.
    out[count == 0] = -np.inf
    return out


def merge_short_segments(
    intervals: np.ndarray,
    labels: list[str],
    min_duration: float = DEFAULT_MIN_DURATION_S,
) -> tuple[np.ndarray, list[str]]:
    """Absorb segments below `min_duration` into the longer neighbour.

    Runs after smoothing, not instead of it. Smoothing removes flicker by
    changing what the detector believes; this removes the leftovers by editing
    the output, which is a cruder operation and is kept separate so the two
    are never confused when reading a result.

    A dropped segment's time is given to whichever neighbour is longer, which
    biases towards sustained chords -- the right bias for solo piano, and a
    questionable one for fast changes. If it ever starts mattering musically it
    should be replaced by the HMM's duration model rather than tuned.
    """
    if len(labels) == 0:
        return intervals, labels

    spans = [[float(s), float(e), lab] for (s, e), lab in zip(intervals, labels)]
    changed = True
    while changed and len(spans) > 1:
        changed = False
        durations = [e - s for s, e, _ in spans]
        i = int(np.argmin(durations))
        if durations[i] >= min_duration:
            break
        if i == 0:
            target = 1
        elif i == len(spans) - 1:
            target = len(spans) - 2
        else:
            target = i - 1 if durations[i - 1] >= durations[i + 1] else i + 1
        lo, hi = min(i, target), max(i, target)
        spans[lo] = [spans[lo][0], spans[hi][1], spans[target][2]]
        del spans[hi]
        changed = True

    # Adjacent segments can now share a label; collapse those too.
    merged: list[list] = []
    for s, e, lab in spans:
        if merged and merged[-1][2] == lab:
            merged[-1][1] = e
        else:
            merged.append([s, e, lab])

    out = np.array([[s, e] for s, e, _ in merged], dtype=float)
    return out, [lab for _, _, lab in merged]


def recognize(
    y: np.ndarray,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop: int = HOP,
    smoothing: int = DEFAULT_SMOOTHING_FRAMES,
    causal: bool = True,
    no_chord_threshold: float = 0.0,
    min_duration: float = DEFAULT_MIN_DURATION_S,
    gamma: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Audio in, labelled chord segments out.

    `gamma=None` -- no log compression -- is the default *here* and not in
    `chroma`, deliberately. Compression as currently ordered is applied to raw
    power in the tens of thousands, where log1p(100*x) is effectively log(x)
    and flattens the contrast the templates need; measured on a synthetic C
    major it takes the winner's margin over the runner-up from 0.253 to 0.023.
    Changing `chroma`'s default would silently move every number in the
    existing tests and the README, so the choice is made at the call site and
    the front end is left alone until the sweep settles it.
    """
    X = chroma(y, sr, n_fft=n_fft, hop=hop, gamma=gamma)
    if X.shape[0] == 0:
        return np.zeros((0, 2)), []

    scores = emission_scores(X, no_chord_threshold=no_chord_threshold)
    smoothed = smooth_scores(scores, window=smoothing, causal=causal)
    states = smoothed.argmax(axis=1)

    intervals, labels = states_to_intervals(
        states, frame_intervals(len(states), sr, hop, n_fft)
    )
    return merge_short_segments(intervals, labels, min_duration=min_duration)


def confidence(
    y: np.ndarray,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop: int = HOP,
    smoothing: int = DEFAULT_SMOOTHING_FRAMES,
    causal: bool = True,
    gamma: float | None = None,
) -> float:
    """Mean gap between the winning chord and the runner-up, over live frames.

    Reported alongside every result because WCSR alone hides the thing that
    actually predicts failure on real audio. A detector can be right on every
    frame with a margin of 0.02 and be one room reflection away from being
    wrong on all of them, and that state of affairs should be visible before it
    turns into a bug report.
    """
    X = chroma(y, sr, n_fft=n_fft, hop=hop, gamma=gamma)
    if X.shape[0] == 0:
        return 0.0
    S = smooth_scores(emission_scores(X), window=smoothing, causal=causal)
    live = np.isfinite(S).all(axis=1)
    if not live.any():
        return 0.0
    top2 = np.sort(S[live], axis=1)[:, -2:]
    return float(np.mean(top2[:, 1] - top2[:, 0]))


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Read a file to mono float at `sr`.

    Audio I/O is the one place a library is unambiguously the right answer --
    decoding wav/mp3/m4a is not what this project is about, and none of it
    crosses into the Rust core, which will be handed samples by the browser.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "reading audio files needs the eval extra: "
            "pip install -e 'python[eval]'"
        ) from exc

    y, file_sr = sf.read(path, dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        # Linear resampling. Adequate for a diagnostic entry point; if
        # resampling ever lands on the measurement path it needs a real
        # polyphase filter, because naive interpolation aliases.
        n = int(round(len(y) * sr / file_sr))
        y = np.interp(
            np.linspace(0, len(y) - 1, n), np.arange(len(y), dtype=np.float64), y
        )
    peak = np.max(np.abs(y)) if len(y) else 0.0
    return y / peak if peak > 0 else y


def _print_timeline(intervals: np.ndarray, labels: list[str], margin: float) -> None:
    if len(labels) == 0:
        print("  no frames -- file shorter than one analysis window?")
        return
    print(f"  {'start':>7} {'end':>7} {'dur':>6}  chord")
    print("  " + "-" * 34)
    for (s, e), lab in zip(intervals, labels):
        bar = "#" * min(int((e - s) * 8), 24)
        print(f"  {s:7.2f} {e:7.2f} {e - s:6.2f}  {lab:<7} {bar}")
    print()
    print(f"  {len(labels)} segments, {transitions_per_minute(intervals):.0f} changes/min")
    print(f"  mean margin over runner-up: {margin:.3f}")
    if margin < 0.05:
        print("  ^ thin. the winner is barely winning; treat these labels as soft.")


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", nargs="?", help="audio file (wav/flac/aiff/ogg)")
    p.add_argument("--demo", action="store_true", help="run on a synthetic progression")
    p.add_argument("--smoothing", type=int, default=DEFAULT_SMOOTHING_FRAMES,
                   help=f"frames in the moving average (default {DEFAULT_SMOOTHING_FRAMES}, 1 disables)")
    p.add_argument("--centred", action="store_true",
                   help="centre the smoothing window (offline only; not real-time honest)")
    p.add_argument("--gamma", type=float, default=None,
                   help="log compression constant (default off; see recognize docstring)")
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    args = p.parse_args(argv)

    if args.demo or not args.path:
        from .signals import progression
        chords = [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")]
        y, ref = progression(chords, seconds_each=2.0, sr=SR)
        source = "synthetic C - Am - F - G"
        print(f"\n{source}")
        print("synthetic audio: sustained, perfectly stable, no room. real piano "
              "will be harder.\n")
    else:
        y = load_audio(args.path, SR)
        print(f"\n{args.path}  ({len(y) / SR:.1f}s)\n")

    kw = dict(smoothing=args.smoothing, causal=not args.centred, gamma=args.gamma)
    intervals, labels = recognize(y, SR, min_duration=args.min_duration, **kw)
    _print_timeline(intervals, labels, confidence(y, SR, **kw))

    window_ms = args.smoothing * HOP / SR * 1000
    if not args.centred and args.smoothing > 1:
        print(f"  causal {args.smoothing}-frame window: answers settle "
              f"{window_ms:.0f} ms after a chord change.")
    print()


if __name__ == "__main__":
    main()
